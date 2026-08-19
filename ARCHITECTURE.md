
# Architecture

What each file does, how a TIR moves through the system, and why the parts that
look unusual are the way they are.

For running the thing, see **[README.md](README.md)**.

---

## The path a TIR takes

```
   QPS export (.xlsx)  ─ any of three column layouts
            │
            ▼
   data.canonicalize        rename every recognised column to one canonical name
            │
            ▼
   data.build_text_series   Description 1 + Description 2 + Doc Title -> one string
            │
   ┌────────┴─────────┐
   │                  │
TRAINING           PREDICTION
   │                  │
   ▼                  ▼
preprocess         models.build_features      word + character TF-IDF
   │                  │
   ▼                  ▼
train              inference.classify         parent first, then its children
   │                  │
   ▼                  ▼
models/            pred / confidence / review flag per field
```

The split matters: **training and prediction compose their input through the
same two functions**, so a TIR typed into the web app and the same TIR in an
uploaded file cannot be read differently.

---

## Modules

### `config.py` — settings, read once

Owns `ROOT` and every path derived from it, and parses `config.json` a single
time. Four modules used to parse it independently and eight re-derived the same
values from it, which is how a setting comes to mean one thing during training
and another at prediction time.

`ROOT` is resolved from the directory name rather than assumed to be one level
up, so the same code runs whether the modules sit under `src/` or flat at the
top level.

Also holds `review_threshold(target)` and `target_title(target)` — the per-field
threshold and the human-facing name, both read from config rather than written
out anywhere.

### `data.py` — reading an export

Two jobs that belong together because both are about turning a raw file into
something the rest of the system recognises.

**Column recognition.** `resolve_columns` matches each canonical field against
the alias list in `config.json`, folding case, underscores and repeated spaces
so `DESCRIPTION_ONE`, `Description One` and `description one` all resolve
without being listed separately. `canonicalize` renames them and leaves every
unclaimed column untouched, so an uploaded file keeps the fields it arrived
with.

**Text.** `build_text_series` joins the configured `text_columns`, skipping any
the file does not carry — the three export layouts do not all include a
Description 2, and a file missing one should still classify on what it has.

`clean_text` collapses whitespace before anything else sees it. Character
n-grams see a doubled space, and 2 % of the sample export contains one.

### `models.py` — features and arithmetic

`build_features` produces the word and character TF-IDF the classifiers read,
transforming a whole column in one call. It used to take a single string, so
classifying a file meant one transform per row and stacking 90,000 matrices.

`ensemble_score_matrix` weights and sums each model's probabilities. Every
member contributes a distribution, so the weighted sum is itself a probability.

> The previous form one-hot encoded the SVM's hard prediction at a flat 0.40,
> which put a floor of 0.40 under the winner and a ceiling of 0.60 over every
> rival. The number reported as "confidence" could not be compared against a
> probability at all, which is why `review_threshold` did not mean what it
> looked like it meant.

`top_k_from_scores` returns the best few labels per row, used where a single
answer would overstate what the data supports.

`verify_model_hash` checks an artifact's SHA-256 **before** it is loaded, since
loading one unpickles and therefore executes.

### `preprocess.py` — building the training data

Canonicalizes **each file individually before concatenating**. The layouts name
the same field three different ways, so combining first would produce a frame of
disjoint, mostly-empty columns.

De-duplicates on the canonical text plus the labels. A whole-row comparison
found nothing between the two exports despite one being 99.4 % contained in the
other, because they disagree on every column name.

Separates *blank* from *unrecognised* before normalising. Both used to land in
the `OTHER` bucket, which turned 30,000 uncoded records into a fabricated
category holding a third of the training rows. Blankness is tested with `isna`
**and** against the placeholder spellings, because pandas 2 renders a missing
value as the string `"nan"` under `astype(str)` while pandas 3 keeps it missing.

Writes `data/hierarchy.json` from the parent/child pairs observed in the
**training split only**, so the validation and test rows cannot influence which
combinations the model may predict. Observed pairs rather than the code prefixes,
which nest in only about 99.9 % of records at the second level and 95.5 % at the
third.

### `train.py` — fitting and choosing

For a field with no parent: fits a calibrated `LinearSVC`, then fits stochastic
gradient, logistic regression and boosting alongside it, keeping any that beats
it on validation macro-F1. Macro rather than accuracy because accuracy sits at
the limit of how consistently the data was coded, so any gain a second model has
to offer is in the rare categories, which only the macro average sees.

Each challenger is fitted inside a guard that records a failure and carries on,
and reports its score the moment it is known — an early run was killed part way
through and threw away two measurements already paid for.

Boosting gets the 30,000 most informative columns by chi-squared rather than all
183,633, with 64 histogram bins. It builds a histogram per feature per class, and
the full width exhausted 13.7 GB and was killed twice. That is less a handicap
than what trees are for: a tree splits on individual features and cannot use a
hundred thousand of them, while a linear model weights all of them at once.

For a field **with** a parent: one classifier per parent value, over that
parent's children only, into `models/<field>/<parent id>/`. Each is fitted on the
*true* parent label; at prediction time the parent is whatever was predicted or
confirmed.

Console output is written for someone deciding whether the models are usable.
The full detail goes to `train.log` through `log()`; `say()` writes the readable
subset to both.

### `inference.py` — the only prediction path

Used by the web app, the batch CLI and the HTTP endpoint, so they cannot drift
apart on feature construction, weighting or the rules for withholding an answer.

`load_bundle` verifies and loads the shared vectorizers and each field's models,
including the per-parent folders and the routing table.

`classify` evaluates fields in config order, which is parent before child.
For each child it groups rows by their parent's answer and routes each group to
that parent's classifier, so the returned code always belongs under the parent.
Four cases are handled explicitly:

| Situation | What happens |
| --- | --- |
| Parent has its own model | Normal prediction inside that parent |
| Parent has exactly one child | That child, confidence 1.0 |
| Parent had too few records to model | Its most common child, flagged for review |
| Parent blank or withheld | Child withheld too |

A blank description is withheld at every level. The models still score an
all-zero vector and can report high confidence doing it — an empty string once
scored 99.8 %.

A parent flagged for review passes that flag down, because a wrong parent costs
the child 9.4 points.

### `feedback.py` — corrections

Corrections are stored in `data/feedback.csv` and used twice: reapplied
immediately by `CorrectionIndex`, and folded into training by `train.py`.

Matching is exact, or by cosine similarity of the same TF-IDF features the
classifiers read. The threshold is 0.80, derived from both directions on the
held-out split rather than inherited: a typo scores 0.815, a plural 0.857 and an
inserted word 0.842, while a reordering drops to 0.610 and a genuine rewording to
0.207. At 0.90 none of those edits match and only an exact repeat is ever
answered.

> This used to use a separate hash-embedding space — every subword mapped to a
> random vector from its SHA-256 digest. Unlike the trained vectors it replaced,
> that carried no relationship between words, and its character 3–5 grams
> duplicated what `tfidf_char` already encodes exactly. Held out it cost 0.1
> accuracy and 0.3 macro-F1 while taking about sixteen times as long to compute
> as training the model.

### `export.py` — the session spreadsheet

Builds the workbook of everything coded in a sitting. Column headings come from
the **first alias of each canonical field**, so what a coder sees and what the
importer accepts cannot drift apart.

### `app.py` — the web app

Three tabs: one TIR at a time, a whole file, and the correction log.

The single-TIR tab asks the coder to confirm or correct the Process Cat *before*
the deeper codes are chosen, and re-predicts them from the answer. That one
control is worth more than every model change tested.

### `classify.py`, `reports.py`, `api.py` — the other entry points

`classify.py` codes a whole file and reports accuracy and macro-F1 wherever the
file carries its own codes. `reports.py` regenerates the two study reports.
`api.py` exposes `POST /predict` for another system.

---

## What lands on disk

| Path | Written by | Tracked |
| --- | --- | --- |
| `data/processed/*.csv` | `preprocess` | no |
| `data/label_map_<field>.json` | `preprocess` | no |
| `data/hierarchy.json` | `preprocess` | no |
| `data/feedback.csv` | the web app | no |
| `models/tfidf_*.pkl` | `train` | no |
| `models/<field>/{svm,sgd,lr}.pkl`, `xgb.json` | `train` | no |
| `models/<field>/ensemble.json` | `train` | no |
| `models/<field>/<parent id>/svm.pkl` | `train` | no |
| `models/<field>/routing.json` | `train` | no |
| `reports/*.md` | `reports` | **yes** |
| `train.log` | `train` | no |

`ensemble.json` records the weights **and the validation scores that justified
them**, so the choice of models can be audited rather than taken on trust.
`routing.json` records each parent's deterministic children and fallbacks.

---

## Extending it

**A new export layout** — add its column names to `column_aliases` in
`config.json`. No code changes.

**A new field to predict** — add an entry to `targets` with its column, minimum
class size and threshold. Give it a `parent` to have it chosen inside another
field, and list it *after* that parent; `config.validate_target_order` fails
loudly otherwise.

**A new model family** — add it to `gate_models` in `train.py` beside the
existing challengers. It needs a `fit` and a `predict_proba`; everything after
that is generic. `save_models` and `inference._load_member_models` key off the
name, so a `.pkl` and an entry in `ensemble.json` are all inference requires.

**Different confidence policy** — `review_threshold` is per field in
`config.json`. `reports.py benchmark` prints the coverage and precision at every
setting, so the trade can be chosen from measurement rather than guessed.

---

## Things that are the way they are for a measured reason

| Looks odd | Why |
| --- | --- |
| One linear model, not an ensemble by default | A calibrated SVM matched the full three-model blend on held-out data. The comparison still runs and keeps anything that beats it. |
| No dense embeddings | The hash embedder measured *worse* than leaving it out, at sixteen times the training cost. |
| Boosting sees only 30,000 columns | The full width exhausted memory and was killed twice. |
| Deep fields are per-parent, not one big model | 543 categories in one model is impractical; ~85 small ones are not, and cannot emit an invalid combination. |
| Macro-F1 decides which model wins | Accuracy is at the ceiling set by coder agreement; the headroom is all in the rare categories. |
| The web app asks for the category first | Worth 9.4 points on the level below it. |
