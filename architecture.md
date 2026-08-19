# Code Map

What every file in this repository is for, and how it works. The README covers
installation and day-to-day use; this covers the code itself.

---

## How the pieces fit together

```
data/raw/*.xlsx
      |
      |  src/preprocess.py      cleans text, normalises categories, splits the data
      v
data/processed/{train,val,test}.csv   +   data/label_map_<target>.json
      |
      |  src/train.py           builds features once, fits three models per target
      v
models/{tfidf_word,tfidf_char,dense_meta}   +   models/<target>/{svm,lr,xgb}
      |
      +---------------+-------------------+
      |               |                   |
 src/ensemble.py   src/api.py    src/streamlit_app.py
   (batch CSV)      (REST)            (web UI)
                                          |
                                          |  reviewer corrections
                                          v
                                   data/feedback.csv  --> folded back in by src/train.py
```

The three inference paths deliberately share `src/utils.py` and `src/embedding.py`
so they cannot drift apart. A test asserts they produce identical predictions.

---

## Repository root

| File | What it is |
| --- | --- |
| `README.md` | Installation, the startup guide, and how to run each tool. Start there. |
| `ARCHITECTURE.md` | This file. |
| `.gitlab-ci.yml` | CI pipeline. Must live at the repository root — GitLab reads it nowhere else. Runs a compile pass, the three check suites, and a config validation on every push. Its `PROJECT_DIR` variable holds the path down to `TIR_Production`. |
| `.gitignore` | Keeps generated material out of version control: `models/`, `data/processed/`, `data/raw/`, label maps, `predictions.csv`, `feedback.csv`, logs and virtual environments. |
| `TIR Export Example1.xlsx` | Sample TIR export, 8,776 records, 56 columns. |
| `TIR Export Example3.xlsx` | **Byte-identical to Example1.** Passing both to preprocessing merges a file with itself; the duplicate guard catches it. |

The briefing deck, `TechNav_Draft_Presentation.pptx`, lives on the `main`
branch rather than here — it is reference material, not part of the pipeline.

---

## Configuration

**`TIR_Production/config/config.json`** — every tunable value in the system, so
none of them are buried in code. Nine blocks:

| Key | Controls |
| --- | --- |
| `random_seed` | Split reproducibility |
| `required_columns` | Which raw columns make up the model's input text |
| `targets` | Which fields to predict, and per field: source column, normalisation table, unknown-value fallback, minimum class size |
| `primary_target` | The field the train/val/test split is stratified on |
| `ensemble_weights` | How the three models are blended |
| `review_threshold` | Confidence below which a row is flagged for a human |
| `embedding` | Vector size, n-gram range, hash seed, cache cap |
| `category_normalization` | Raw category spelling → standard form |
| `feedback` | Whether reviewer corrections apply at prediction time, and the match threshold |

**`TIR_Production/requirements.txt`** — pinned dependencies. Notably no
PyTorch and no FastText: the embedder is self-contained, so nothing is
downloaded at install or run time beyond these packages.

---

## Source — `TIR_Production/src/`

### `paths.py` — 13 lines
Defines `ROOT`, the project directory, as the parent of `src/`. Every other
module derives its paths from it, so nothing hardcodes a location and the code
runs from any working directory.

### `utils.py` — 278 lines
The shared helpers the whole pipeline depends on, kept in one place so the
training and inference paths cannot diverge.

- `clean_text` collapses every run of whitespace to a single space, trims and lower-cases.
- `build_text_series` joins the configured description columns into the model's input, running `fillna` before `astype` so a missing value never becomes the literal string `"nan"`.
- `normalize_categories` maps raw category spellings onto their standard form, used by both preprocessing and metric reporting so they read a file the same way.
- `build_feature_matrix` produces one row's features: word TF-IDF, character TF-IDF, and the dense embedding, concatenated in the order training used.
- `combine_ensemble_scores` blends the three models — the SVM's hard prediction is one-hot encoded, then weighted against the two probability outputs.
- `expand_proba` widens a model's probability columns onto the full label space using its `classes_`, so a category absent from training cannot silently shift the others.
- `is_blank_text` identifies rows with nothing to classify; callers withhold a prediction rather than let the models score an all-zero vector.
- `hstack_csr` / `vstack_csr` narrow SciPy's sparse union type to `csr_matrix`.
- `hash_file` / `verify_model_hash` compute and check the SHA-256 digests that guard model artifacts.

### `embedding.py` — 268 lines
Turns text into a fixed-length vector **without any trained model file**. Each
piece of text is split into character n-grams plus whole words and their
4-character prefixes and suffixes; every one of those subwords is mapped to a
fixed pseudo-random vector derived from its SHA-256 digest, and the text's
embedding is the mean of them. Because the digest is stable across machines and
Python versions, identical text always produces an identical vector — this is
what replaced FastText, and why the project needs no deep learning framework.
`SubwordHashEmbedder.save_meta` records the settings so inference rebuilds a
byte-identical embedder.

### `preprocess.py` — 229 lines
Turns raw exports into training data. Reads the files named on the command line
from `data/raw/`, drops rows repeated across them, builds the input text, then
for each configured target normalises the category column, excludes classes with
too few records, writes `data/label_map_<target>.json` and assigns numeric
labels. Finally splits into train/validation/test, stratified on the primary
target. Rows a target cannot use are marked `-1` rather than dropped, since
another target may still be able to use them.

### `train.py` — 295 lines
Fits the models. Builds the TF-IDF vectorizers and dense embeddings **once** —
they are unsupervised, so every target shares them — then for each target fits a
calibrated LinearSVC, a Logistic Regression and an XGBoost booster on that
target's labelled rows, writing them to `models/<target>/` with a SHA-256 digest
beside each. Calibration folds are derived from the rarest class rather than
fixed, and categories too thin to fit are named and skipped. Reviewer
corrections from `data/feedback.csv` are folded into the training split only, so
validation and test stay honest.

### `ensemble.py` — 153 lines
Batch inference. Takes a CSV or Excel file, builds the feature matrix once, and
scores it against every configured target, adding `pred_`, `confidence_` and
`review_` columns per target. Where the input carries the true category it also
prints accuracy and a per-category report, normalising the truth column the same
way preprocessing did so the comparison is valid. Model paths come from the
config rather than the command line.

### `api.py` — 127 lines
A FastAPI service exposing `POST /predict`. Verifies and loads every artifact at
import time — integrity checks run *before* `joblib.load`, since unpickling
executes — then answers each request with a prediction, confidence and review
flag per target. A blank description returns no classification rather than a
guess.

### `streamlit_app.py` — 521 lines
The web UI, in three tabs. **Single TIR** classifies one description and shows
each target side by side with a correction dropdown. **Batch file** classifies an
upload, showing only the description and each target's category, confidence and
review flag, with a filter for just the rows needing a human and a review-queue
download. **Corrections** lists what reviewers have logged. Starts with a
`sys.path` bootstrap because `streamlit run` executes the file as a script rather
than as part of the `src` package.

### `feedback.py` — 230 lines
Stores and reapplies reviewer corrections. `append_feedback` writes each one to
`data/feedback.csv` tagged with the field it corrects; `feedback_training_rows`
converts them into labelled rows for the next training run. `CorrectionIndex`
answers a lookup with a reviewer's verdict when the text matches one already
corrected — exactly, or by cosine similarity of the dense embeddings above a
calibrated threshold. That runtime override is **off by default**, so a given set
of models always produces the same answer.

### `tune_weights.py` — 157 lines
Grid-searches the ensemble blend against the **validation** split and reports the
best combination; tuning on test would make the reported accuracy optimistic.
Declines to write anything when targets prefer different weightings. Its current
finding is that tuning is not worth doing — the gain is inside noise.

---

## Checks — `TIR_Production/test/`

Run as scripts (`python test/test_embedding.py`), **not** with `python -m
test.…` — the folder name `test` collides with CPython's own stdlib `test`
package. Each file locates the modules itself, so it runs from any directory.

| File | Covers |
| --- | --- |
| `test_embedding.py` | Determinism — including across processes with different `PYTHONHASHSEED` — vector shape and dtype, empty input, cache transparency, seed isolation, metadata round-trip and version rejection. |
| `test_feedback.py` | The correction store: round-trip, latest-wins, exact and near-duplicate matching, threshold handling, per-target filtering, and unknown categories being dropped. |
| `test_inference.py` | A trained pipeline's behaviour: well-formed output, blank input withheld, determinism, batch matching single, case and padding invariance, odd inputs handled without raising, and partial text still classifying. Skips itself when `models/` is empty. |

---

## Generated, not tracked

| Path | Produced by |
| --- | --- |
| `data/processed/{cleaned_for_training,train,val,test}.csv` | `src.preprocess` |
| `data/label_map_<target>.json` | `src.preprocess` |
| `models/` — encoders, per-target classifiers, `.sha256` digests | `src.train` |
| `data/feedback.csv` | The web app, as reviewers correct predictions |
| `train.log` | `src.train` |
| `predictions.csv` | `src.ensemble` or the app's download button |

All of it rebuilds from `src.preprocess` and `src.train`, which is why none of it
is in version control.