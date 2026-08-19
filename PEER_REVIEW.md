
# Peer Review Plan

Companion to **Peer Review Checklist - Code Inspection.xls** (Rev 1, 2012-04-05).

That checklist was written for embedded C and Ada — it asks about null pointer
dereferencing, memory deallocation, interrupt handlers, bus widths and `goto`.
Its own header says *"each project should tailor the checklist for their needs."*
This document is that tailoring, plus the findings already open and an ordered
procedure for working through it.

**Sections 2 and 3 are where a reviewer's time is worth the most.**

---

## 1. The checklist, tailored

### Applies directly

| ID | Item | Where to check it |
| --- | --- | --- |
| CST-1 | Style consistent throughout | All 13 modules in `TIR_Production/src/`, one author |
| CST-3 | Revision marked in the comment header | **See finding F-2 below** |
| COR-1 | Is the logic correct? | §4 procedure, step 3 |
| COR-2 | Checks to ensure data integrity | Translated below |
| COR-3/4 | Implements the requirements it is traced to | **See findings F-1 and F-3** |
| CST-4 | Does the code reflect the design? | `ARCHITECTURE.md` is the design record |
| APP-1 | Sufficiently data-driven | `config/config.json` drives targets, aliases, thresholds. Adding an export layout or a field needs no code |
| APP-5 | Unused/unnecessary files referenced | **See finding F-5** |
| APP-6 | Obvious dead code | 16 modules to 13; `tune_weights.py` and `embedding.py` deleted as unreferenced |
| APP-8 | Errors/exceptions obvious and concise | Check `preprocess` and `inference` raise with the fix in the message |
| APP-10 | Compiles cleanly with warnings enabled | `pyright` — 0 errors. Two warning filters exist and both carry a written justification |
| APP-12 | Redundant or duplicate code | The consolidation commit `825a5c1` exists for this reason |
| APP-13 | Loop endpoints, off-by-one | Highest risk is the label-id round trip, §3 |
| APP-14 | Type conversions and loss of precision | Translated below — this caught a real defect |
| APP-24 | Absolute pathnames hardcoded | **Checked: none.** All paths derive from `config.ROOT` |
| APP-26 | Complexity under ten branches | **See finding F-4 — ten functions fail this** |
| APP-27 | SLOC per subprogram | **See finding F-4** |
| TST-1 | Are the test cases still valid? | 46 tests; check they assert behaviour and not implementation |

### Applies once translated out of C

The literal question does not apply, but the failure mode does — and in two
cases it is where real defects were actually found.

| Checklist asks | For this project, read as | Status |
| --- | --- | --- |
| Checks to prevent dereferencing null pointers | Handling of `None` and pandas `NA` | **Two real defects found here.** `st.selectbox` returns `None` on an empty list; `int()` on a pandas `NA` raises. Both fixed in `d8447c4`. Assume more exist |
| Checks to ensure data is within bounds | Label ids stay inside the label map; a per-parent model's classes map back to the right global ids | `models.expand_proba` — the single most damaging place to be wrong |
| Data from external interfaces is valid before use | A QPS export with unexpected or missing columns | `data.resolve_columns` and `validate_input_dataframe` |
| Is allocated memory properly deallocated; memory leaks | Peak memory on the full matrix | **Was killed twice by the kernel.** Boosting now runs on a reduced feature space, measured at 3.27 GB |
| Type conversions causing unwanted loss | `astype(str)` on a missing value | **Real defect.** pandas 2 renders it `"nan"`, pandas 3 keeps it missing — a check written for one silently did nothing on the other |
| Are all cases present in a select statement | Every branch of the parent/child routing | Four cases in `inference._predict_child`; confirm none is unhandled |
| Unsafe library or system calls | `joblib.load` unpickles, which executes | Guarded: SHA-256 verified **before** load, in `models.verify_model_hash` |

### Does not apply, and why

Record the reason in the checklist rather than leaving the box blank.

| Checklist item | Why not |
| --- | --- |
| Memory allocation and deallocation, leaks | Python is garbage-collected; no manual allocation |
| Interrupt context, interrupt handlers (3 items) | No interrupt-level code |
| Physical device access, bus width | No device access |
| Network byte order, packing, alignment | No socket-level protocol; HTTP and files only |
| `goto` statements | Not in the language |
| Non-standard compiler features | Interpreted; no compiler |
| POSIX conformance, operating-system interfaces | Not OS-level code |
| Elevated permissions | Runs as an ordinary user; writes only inside the project |
| `else-if` where a case statement fits | Dispatch is by dictionary lookup throughout |

---

## 2. Findings already open

Found while preparing this. Raised here so review time goes on what has **not**
been found.

**F-1 — Requirement traceability was lost in the refactor, now restored.**
The prototype traced seven requirements. Splitting `utils.py` into `data.py` and
`models.py`, and `paths.py` into `config.py`, carried the code but not the
markers — REQ-001, 002, 010, 011 and 012 lost their trace. Restored. Verify:

```bash
for r in 001 002 010 011 012 014; do grep -rl "REQ-$r" TIR_Production/src/; done
```

**F-2 — No module carries an author or revision header.** CST-3 asks for this.
The prototype had them on two modules; none of the current 13 has one. Not
fixed, because the convention should be the project's rather than mine. **Decide
what CST-3 requires here and I will apply it.**

**F-3 — REQ-013 has no implementing code.** *"Deterministic dense text
embeddings, no external model files."* The module was deleted: measured against
a held-out split it cost 0.1 accuracy and 0.3 macro-F1 while taking about
sixteen times as long to compute as training the model. **A requirement was
retired by measurement, which is a decision the SDD should record rather than
something a code review absorbs silently.**

**F-4 — Ten functions exceed the complexity limit; ten exceed a reasonable
length.** APP-26 asks for branches of control under ten.

| Function | Complexity | Lines |
| --- | ---: | ---: |
| `inference.classify` | 19 | 88 |
| `app.render_single` | 19 | 100 |
| `preprocess.main` | 17 | 124 |
| `inference.load_bundle` | 17 | — |
| `reports.build_sufficiency` | 17 | 141 |
| `train.main` | 16 | 166 |
| `inference._predict_child` | 15 | 66 |
| `reports.build_benchmark` | 14 | 114 |
| `train.train_hierarchical` | 14 | 100 |
| `train.gate_models` | 11 | 157 |

Average across all 81 functions is 5.2, so this is concentrated rather than
general. The `main` functions are orchestration, which is a weaker case for
concern than tangled logic. **`inference.classify` is not** — it is the highest
complexity, and independently the function I would look at first for a defect.
That convergence is worth taking seriously.

**F-5 — `api.py` is referenced by nothing.** APP-5. It is a working HTTP
endpoint with no known caller. Either something is going to call it, or it and
four pinned dependencies should go.

---

## 3. Where a defect would do the most damage

Ranked by consequence, not likelihood.

1. **`inference.classify` and `_predict_child`** — rows are grouped by their
   parent's predicted label and each group routed to that parent's model. A row
   in the wrong group gets a confidently wrong answer and raises nothing.
2. **`models.expand_proba`** — scatters a per-parent model's local classes into
   the global label space. Off by one here produces plausible, systematically
   wrong codes.
3. **`preprocess.build_hierarchy`** — built from the **training split only**. If
   validation or test rows leaked in, every hierarchical figure is flattered.
4. **`feedback.CorrectionIndex`** — a wrong match applies a reviewer's label to
   a TIR they never saw, carrying their authority.
5. **`data.canonicalize`** — two columns folding to one canonical name. There is
   a guard; confirm it holds.

---

## 4. Procedure

**Step 1 — Read the commit log (20 min).**

```bash
git log --oneline origin/main..HEAD
```

Sixteen commits, each carrying the measurement that motivated it. The messages
are the design record; read them before any diff.

**Step 2 — Reproduce the numbers (30 min, mostly waiting).**

```bash
cd TIR_Production
python -m pytest test                    # 46 tests, ~2s
python -m src.preprocess --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"
python -m src.train --no-gate            # ~7 min
python -m src.reports benchmark
```

Expect accuracy within a point of 92.8 / 88.2 / 80.8 / 73.6. **A number that
does not reproduce is a finding — raise it.**

**Step 3 — Read `src/inference.py` in full (30 min).** 292 lines, and every
prediction goes through it. This is where §3 says the damage is.

**Step 4 — Work §1 against the code (45 min).** The rows marked *"see finding"*
are done; the rest need a reviewer.

**Step 5 — Decide on §5 (15 min).** These are judgement calls, and disagreeing
is a legitimate outcome.

---

## 5. Judgement calls open to challenge

| Call | Grounds | Push back if |
| --- | --- | --- |
| Correction reuse at 0.80 similarity | Typo 0.815, plural 0.857, inserted word 0.842; reordering 0.610, rewording 0.207 | Reusing across a plural is too loose for you |
| Level 3 returns a ranked three, not one answer | 67 % coder agreement | You would rather have one answer and a flag |
| `min_class_size` 10 for the deep fields | Keeps 96.7 % of records | Rare codes should appear even if unlearnable |
| Boosting limited to 30,000 features | Full width was killed by the kernel twice | It handicaps boosting unfairly |
| Kept models blended with **equal weights** | No held-out evidence for a split | **This is a real gap.** SGD scored 73.5 against the SVM's 71.1 on Process Cat and gets the same 50 % |
| The deep fields never get a challenger | Not a decision — an omission | **Also a real gap.** `train.py:466` calls `fit_svm` directly; two of four fields are SVM-only by construction |
| Coder agreement grouped on Description 1 alone | The full model input shrinks the comparable set tenfold | You think the grouping inflates disagreement — this moves every ceiling figure |

---

## 6. What is unverified

- **Windows.** Preprocess and train have run there and matched. The Streamlit
  app, the session export and the HTTP endpoint have not.
- **The export opening in Excel proper.** Checked by reading it back with
  pandas, not by opening it.
- **The app end to end.** Streamlit is not installed in my environment — which
  is exactly why the two defects your editor caught were invisible to me.
- **Anything about how coding actually works today.** All inferred from data.

---

## 7. My error patterns

A defect still present is more likely to be one of these shapes than a novel one.

| What happened | The pattern |
| --- | --- |
| Called the memory problem fixed after measuring only matrix construction. Killed again. | Measuring part of a path, reporting on the whole |
| Set the correction threshold from false-positive data alone; a typo then failed to match | Optimising one side of a trade-off |
| Dropped the `selectbox`-returns-`None` guard while consolidating | Losing a defence in a refactor because its reason was in a comment I removed |
| Lost REQ traceability in the same refactor (F-1) | The same pattern again, which is why F-1 is worth checking for elsewhere |

---

## 8. Questions to put to me

- Why is coder agreement grouped on Description 1 rather than the full input?
- Why does the hierarchy come from observed pairs rather than the code prefixes,
  when the codes clearly nest?
- What happens on an export whose layout no alias covers — loud or quiet?
- Boosting scored worst on every field. The method, or the feature limit?
- Is `inference.classify` at complexity 19 worth decomposing, or is the routing
  genuinely irreducible?

`ARCHITECTURE.md` covers what each module does. `README.md` covers running it.
