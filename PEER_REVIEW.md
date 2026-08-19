
# Reviewing This Work

Fifteen commits and about 3,100 lines of source. The prototype they replace is
commit `7b95865` — `git show 7b95865:<file>.py` reads any of it, since the files
were flat at that point. Reading all of it is the least
efficient way to review it.

Almost every decision here was made because a measurement said so. That makes
the work easy to check and easy to falsify: for each claim below there is a
command that either reproduces the number or contradicts it. **If a number
below does not reproduce, that is a finding — please raise it.**

Start with §1 and §5. Those are where a reviewer's time is worth the most.

---

## 1. The four claims everything else rests on

If these are wrong, much of the rest is wrong with them.

### Claim: the two exports overlap, and de-duplication catches it

> 12,321 duplicate records; the smaller export is 99.4 % contained in the larger.

```bash
cd TIR_Production
python -m src.preprocess --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"
```

Look for `Removed 12,321 duplicate row(s)`. To check the overlap independently:

```python
import pandas as pd
from src.data import canonicalize
a = canonicalize(pd.read_excel("../QPS Pull ….xlsx"))
b = canonicalize(pd.read_excel("../TIR Export Example1.xlsx"))
key = lambda d: set(d.description_1.astype(str).str.strip().str.lower())
print(len(key(a) & key(b)) / len(key(b)))     # expect ~0.994
```

**Why it matters:** if this is wrong, every accuracy figure is inflated, because
the same TIR would sit in both the training data and the data used to judge it.

### Claim: blanks were being learned as a real category

> A third of records have no Process Cat; they were becoming `OTHER`.

```bash
git show 7b95865:preprocess.py | sed -n '/def prepare_target/,/^# /p'
```

The original passed `unknown_value` into `normalize_categories`, which
`fillna`s **every** unmatched value including genuine blanks. Confirm the
current behaviour reports `process_cat: 27 categories … (67.4 %)` rather than
28 categories at 100 %.

### Claim: coders disagree, and that is the ceiling

> 85.3 % agreement on Process Cat, 67.0 % on Level 3.

```bash
python -m src.reports sufficiency --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"
cat reports/data_sufficiency.md
```

**Challenge this one hardest.** It is a lower bound computed from records
sharing a Description 1, and two such records can legitimately be different
events. The measure lives in `src/reports.py: consistency`, and
`test/test_reports.py` pins it to a hand-checked case. If you think the grouping
is wrong, that is the single most consequential disagreement you could raise —
every accuracy figure is reported against it.

### Claim: confidence is now a probability

> The old blend put a floor of 0.40 under the winner.

```bash
git show 7b95865:utils.py | sed -n '/def combine_ensemble_scores/,/return combined/p'
```

The SVM's hard prediction was one-hot encoded and weighted 0.40, so the winning
score could not fall below 0.40 nor a rival exceed 0.60. Current behaviour:

```bash
python -m pytest test/test_models.py -k probability -v
```

**Why it matters:** every "auto-code X % at Y % precision" statement depends on
this. If confidence is not a probability, those statements mean nothing.

---

## 2. Reproduce the headline numbers

```bash
python -m pytest test                    # 46 tests, ~2s
python -m src.preprocess --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"
python -m src.train --no-gate            # ~7 min, SVM only
python -m src.reports benchmark
```

`reports/benchmark.md` should give accuracy within about a point of 92.8 / 88.2 /
80.8 / 73.6, and Process Cat coverage near 80 % at 95 % precision.

Run without `--no-gate` (hours) to reproduce the model comparison:

| Field | SVM | SGD | Logistic | Boosting |
| --- | --- | --- | --- | --- |
| Metric Cat | **73.0** | 65.8 | 66.3 | 67.8 |
| Process Cat | 71.1 | **73.5** | 70.6 | 65.4 |

Splits are seeded, so these should land within a point. **A larger gap is worth
asking about.**

---

## 3. Where I would look for bugs first

Ranked by how much damage a defect would do, not by how likely it is.

1. **`inference.classify` — the per-parent routing.** `src/inference.py`. Rows
   are grouped by their parent's predicted label and each group routed to that
   parent's model. A row landing in the wrong group gets a confidently wrong
   answer with no error anywhere. Check the four branches in `_predict_child`:
   parent has a model, parent has one child, parent has no model, parent blank.
2. **`preprocess.build_hierarchy`.** Built from the **training split only**. If
   validation or test rows leaked in, the model would be allowed combinations it
   was judged on — a subtle leak that would flatter every hierarchical number.
3. **The label-id round trip.** `label_map_*.json` maps id to name and several
   places invert it. An off-by-one would produce plausible, systematically wrong
   codes. `models.expand_proba` is where a per-parent model's local classes are
   scattered back into the global label space — the most likely place to get it
   wrong.
4. **`feedback.CorrectionIndex`.** A wrong match applies a reviewer's label to a
   TIR they never saw, carrying their authority. Threshold 0.80, calibrated in
   §4 below.
5. **`data.canonicalize` collisions.** Two columns folding to the same canonical
   name. There is a guard; check it holds for a file with both `Description 1`
   and `DESCRIPTION_ONE`.

---

## 4. Judgement calls, not facts

These are defensible but arguable. Disagreeing with them is a legitimate review
outcome.

| Call | Reasoning | Push back if |
| --- | --- | --- |
| Correction reuse at **0.80** | Typo scores 0.815, plural 0.857, inserted word 0.842; reordering 0.610, genuine rewording 0.207 | You think reusing a correction across a *plural* is too loose |
| Level 3 shows **top 3**, not one answer | 67 % coder agreement; a single answer overstates the evidence | You would rather have one answer and a review flag |
| `min_class_size` **10** for Sub and Level 3 | Keeps 96.7 % of records; below it a class cannot be learned | You want rare codes represented even if unlearnable |
| Boosting limited to **30,000 features** | Full width exhausted memory and was killed twice | You think that handicaps it unfairly against the linear models |
| Kept models blended with **equal weights** | No held-out evidence for a better split | **Agree with you — this is a real gap.** SGD scored higher than the SVM on Process Cat and gets the same 50 % |
| Deep fields never get a challenger | Not a decision, an omission | **Also a real gap.** `train.py:466` calls `fit_svm` directly; two of four fields are SVM-only by construction |

The last two rows are the strongest criticisms available of this work, and they
are mine rather than yours only because I found them while writing this.

---

## 5. What I could not verify

Please treat these as unproven.

- **Windows.** Everything here ran on Linux. You have run preprocess and train
  on Windows and the numbers matched, but the Streamlit app, the session export
  and the HTTP endpoint have not been exercised there.
- **The export opening in Excel proper.** `src/export.py` was checked by reading
  it back with pandas, not by opening it in Excel.
- **The app end to end.** Streamlit is not installed in my environment. The
  logic is covered by tests; the interface is not. Two of the defects your
  editor caught — `st.selectbox` returning `None`, and `int()` on a pandas `NA`
  — were invisible to my checker for exactly this reason.
- **Whether the 30,000-feature limit costs boosting anything.** It scored worst
  everywhere, but that may be the limit rather than the method.
- **Any claim about your coding process.** Everything about how TIRs are coded
  today is inferred from the data. The twelve questions in the session document
  exist because of this.

---

## 6. Mistakes I made in this work

Not confession — pattern. These are the kinds of error most likely to still be
present somewhere I have not looked.

| Mistake | The pattern behind it |
| --- | --- |
| Called the memory problem fixed after measuring only matrix construction, not training. It was killed again. | **Measuring part of a path and reporting on the whole path.** |
| Set the correction threshold to 0.90 from false-positive data alone. A one-letter typo then failed to match — the feature's whole purpose. | **Optimising one side of a trade-off and not checking the other.** |
| Dropped the `st.selectbox`-returns-`None` guard while consolidating modules. | **Losing a defence during a refactor because its reason was in a comment I removed.** |
| Claimed 15 modules would become 9. It was 13. | **Quoting a number before doing the work.** |

If you find a defect, it is more likely to be one of these shapes than a novel
one.

---

## 7. Questions worth putting to me

- Why is the coder-agreement measure grouped on Description 1 alone rather than
  the full model input? *(Answer: the full input shrinks the comparable set
  tenfold. But it is a judgement call and it moves every ceiling figure.)*
- Why does the hierarchy come from observed pairs rather than the code prefixes,
  when the codes clearly nest?
- What happens on a QPS export with a column layout none of the aliases cover —
  does it fail loudly or quietly?
- Why is `api.py` still here if nothing calls it?
- Boosting scored worst on every field. Is that the method or the feature limit?

---

## 8. Reviewing efficiently

Two hours, in this order:

1. `git log --oneline origin/main..HEAD` — fifteen commits, each with the
   measurement that motivated it in the message. The messages are the design
   record; read them before the diffs.
2. Run §2. If the numbers reproduce, the pipeline works end to end.
3. Read `src/inference.py` in full — 292 lines, and every prediction the system
   makes goes through it.
4. Read §4 and §6 and decide whether you agree.
5. Skim the rest.

`ARCHITECTURE.md` explains what each module does. `README.md` covers running it.
