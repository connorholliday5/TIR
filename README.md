
# TIR Classification System

The TIR Classification System automatically analyzes text from Technical Issue Reports (TIRs) and assigns each report to the correct category. This reduces manual review time and improves routing and understanding of technical issues.

The entire pipeline is automated: once raw data is provided, the system cleans text, builds embeddings, trains the models, and prepares prediction tools.

---

## How the System Works (Non‑Technical Overview)

### 1. Raw Data
Place exported TIR CSV or Excel files into:
```
data/raw/
```
These contain the text the system learns from.

### 2. Preprocessing
The preprocessing pipeline (`src/preprocess.py`):
- Removes unwanted symbols, spacing, and formatting
- Normalizes text
- Combines description fields
- Normalizes category labels
- Splits data into train/validation/test sets

All cleaned outputs are written to:
```
data/processed/
```
Embeddings are not built here — they are derived from the cleaned text during
training.

### 3. Model Training
The training pipeline creates several types of text features:
- TF‑IDF word features (1–2 grams)
- TF‑IDF character features (3–5 grams)
- Deterministic dense subword embeddings

#### Text Embeddings
Dense embeddings come from `src/embedding.py` (`SubwordHashEmbedder`) — a
self-contained embedder that replaces the previous FastText vectors. It needs
no pretrained model file, no training corpus of its own, and no deep learning
framework: **numpy is the only dependency, and neither PyTorch nor FastText is
installed anywhere in this project.**

How it works: each text is split into character n‑grams plus whole words and
their 4‑character prefixes/suffixes. Every one of those subwords is mapped to a
fixed pseudo‑random vector derived from its SHA‑256 digest, and the text's
embedding is the mean of those vectors. Because the digest is stable across
machines, processes and Python versions, identical text always produces an
identical vector — no `PYTHONHASHSEED` sensitivity, no model file to ship.

Settings live in the `embedding` block of `config/config.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `dim` | `192` | Length of each embedding vector |
| `min_n` / `max_n` | `3` / `5` | Character n‑gram sizes |
| `seed` | `0` | Namespace for the subword hashes; change it for a fresh, uncorrelated embedding space |
| `l2_normalize` | `false` | Scale vectors to unit length (helps when texts vary a lot in length) |

Training writes the resolved settings to `models/dense_meta.json`, and
`api.py`, `ensemble.py` and `utils.build_feature_matrix` rebuild the identical
embedder from that file — so inference can never silently disagree with
training. **Changing any of these values requires retraining** (`python -m
src.train`), because the feature columns change.

Sanity checks for the embedder:
```
python test/test_embedding.py
```

Models trained:
- Calibrated Support Vector Machine
- Logistic Regression
- XGBoost Booster

Model files are saved under:
```
models/
```

### 4. Weighted Ensemble
The TIR Classification system uses a weighted ensemble combining predictions from Support Vector Machine (SVM), Logistic Regression, and XGBoost. Each model contributes differently based on predefined weights stored in `config/config.json`. This ensemble improves accuracy, stability, and robustness across diverse TIR texts.

### 5. Prediction Tools
Once trained, TIRs can be classified in batch (`src.ensemble`) or one at a time
through the FastAPI service. See **Step 8** of the Startup Guide below for the
commands.

---

## Project Structure

The repository root holds the briefing material and the CI definition; the
project itself lives in `TIR_Production/`.

For what each file does and how it works, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
.gitignore
.gitlab-ci.yml                 Must sit here, at the repository root
README.md
TIR Export Example1.xlsx       Sample data
TIR Export Example3.xlsx       Identical to Example1 - see the note in Step 6
ARCHITECTURE.md                What each file does and how it works

TIR_Production/
  config/config.json           Configuration: targets, weights, embedding, thresholds
  data/raw/                    Raw TIR files (you create this; not tracked)
  data/processed/              Cleaned datasets, written by src.preprocess
  data/label_map_*.json        One per target: label ID -> category name
  data/feedback.csv            Reviewer corrections logged from the web app
  logs/
  models/                      Shared encoders, plus one folder of models per target
  requirements.txt
  src/preprocess.py            Cleaning, labelling, and train/val/test splitting
  src/train.py                 Training pipeline
  src/ensemble.py              Batch inference pipeline
  src/api.py                   FastAPI service
  src/streamlit_app.py         Streamlit web app
  src/embedding.py             Deterministic dense text embeddings
  src/feedback.py              Reviewer corrections: storage, lookup, retraining rows
  src/utils.py                 Shared utilities
  src/tune_weights.py          Grid-searches the ensemble weights on validation data
  src/paths.py                 Project-wide path constants
  test/test_embedding.py       Embedder sanity checks
  test/test_feedback.py        Correction store and lookup checks
  test/test_inference.py       Trained-pipeline behaviour (skips without models)
  train.log                    Training log
```

Everything under `models/`, `data/processed/`, `data/raw/` and the label maps
is generated - `src.preprocess` and `src.train` rebuild all of it, so none of
it is tracked.

---

## Startup Guide

Start here if you have just cloned the repository. Steps 1–7 take you from an
empty checkout to working predictions; run them in order, from the repository
root, in a single terminal session.

**Prerequisites:** Python 3.11 or newer (the pinned `numpy==2.4.3` requires it —
check with `python --version`) and git. No GPU, no model downloads, and no
network access is needed after step 3.

---

### 1. Clone the Repository
```
git clone <repository-url>
cd <repo>/TIR_Production
```
Every command below is run from `TIR_Production`, not the repository root -
that is where `config/`, `src/` and `requirements.txt` live. They are written as
`python -m src.<module>` rather than `python src/<module>.py`, because the
modules import each other through the `src` package — running the file
directly gives `ModuleNotFoundError: No module named 'src'`.

---

### 2. Create a Virtual Environment

#### Mac/Linux
```
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows
```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies
```
pip install -r requirements.txt --no-cache-dir --index-url https://artifact1.ebnet.gdeb.com/artifactory/api/pypi/d604.seg.python.external/simple --trusted-host artifact1.ebnet.gdeb.com
```
On a machine with ordinary PyPI access, `pip install -r requirements.txt` is
enough.

---

### 4. Create the Data Folders
```
mkdir -p data/raw          # Windows: mkdir data\raw
```
Git does not track empty folders, so a fresh clone may not have these. The
pipeline creates `data/processed/` and `models/` itself.

---

### 5. Add Raw TIR Data
Copy your exported TIR files (`.xlsx`, `.xls` or `.csv`) into `data/raw/`:
```
cp "TIR Export Example1.xlsx" data/raw/
```
Each file must contain **`Description 1`** (the text the models read) plus the
source column of every configured target — **`Process Cat`** and
**`Metric Cat`** by default. Any that are missing are reported by name before
the run starts. Every other column is carried through untouched and ignored.

The input columns are set by `required_columns` in `config/config.json`, and
listing more than one joins them into a single text field. `Description 2` was
dropped from that list: it is empty in 8,775 of 8,776 rows of the sample
export and measured worth +0.02 % accuracy.

**Which fields get predicted** is set by the `targets` block of
`config/config.json`. Each entry names the column to learn:

```json
"primary_target": "process_cat",
"targets": {
  "process_cat": { "column": "Process Cat", "normalization": "category_normalization",
                   "unknown_value": "OTHER", "min_class_size": 3 },
  "metric_cat":  { "column": "Metric Cat", "min_class_size": 3 }
}
```

| Key | Meaning |
| --- | --- |
| `column` | The raw export column holding the category |
| `normalization` | Name of a map in `config.json` that standardises the values; omit when the column is already consistent |
| `unknown_value` | Category for values the map does not cover |
| `min_class_size` | Categories with fewer records are not trained, and are named when preprocessing runs |

`primary_target` is the field the train/validation/test split is stratified on.
Adding a target is a config change: preprocess, train, batch inference, the API
and the UI all pick it up. Each target is trained independently — the roll-up
from a deeper level was measured 3–4 points worse than predicting each level
directly.

---

### 6. Preprocess the Raw Data
```
python -m src.preprocess --raw_csv "TIR Export Example1.xlsx"
```
`--raw_csv` takes **file names, not paths** — each is looked up inside
`data/raw/`. Pass several to merge them into one dataset:
```
python -m src.preprocess --raw_csv "March export.xlsx" "April export.xlsx"
```

> Do not pass both shipped samples. `TIR Export Example3.xlsx` is byte for byte
> identical to `Example1.xlsx`, so merging them contributes nothing. The run
> reports what it removed — `Merged 2 files. Total rows: 17552` followed by
> `Removed 8,776 duplicate row(s)` — and proceeds on the 8,776 real records.
> Without that guard the same TIR would land in both the training and the test
> split, and the reported accuracy would be inflated by rows the model had
> already seen.

This cleans the text, normalises the category names, and splits the data into
roughly 72 % train / 13 % validation / 15 % test, stratified by category.
Afterwards you will have:

| File | Contents |
| --- | --- |
| `data/processed/cleaned_for_training.csv` | Every row, cleaned and labelled |
| `data/processed/train.csv` | Training split |
| `data/processed/val.csv` | Validation split (early stopping) |
| `data/processed/test.csv` | Held-out test split |
| `data/label_map_<target>.json` | Numeric label ID → category name, one file per target |

Takes a few seconds for ~9k rows.

> **Rerun preprocessing and training together.** Each `label_map_<target>.json` is rebuilt
> from whatever data you feed in, and the IDs are positional — so a label map
> from one run will silently disagree with models from another.

---

### 7. Train the Models
```
python -m src.train
```
This builds the TF‑IDF vectorizers and the dense embeddings, then trains the
SVM, Logistic Regression and XGBoost models. **Expect this to run for a while**
— measured at about 12 minutes for 8,776 rows across 20 categories on a 4‑core
machine, nearly all of it in the calibrated SVM and XGBoost. There is no
progress bar and long gaps between lines are normal; `Training complete.` is
the last line.

Afterwards `models/` contains the shared encoders plus a folder per target:
```
models/
  tfidf_word.pkl   tfidf_char.pkl   dense_meta.json      shared by every target
  process_cat/     svm.pkl  lr.pkl  xgb.json
  metric_cat/      svm.pkl  lr.pkl  xgb.json
```
plus a `.sha256` digest beside each one, which inference checks before loading.
The vectorizers and embeddings are unsupervised, so they are built once and
reused; only the classifiers are trained per target.
A `train.log` is written at the repository root.

Note that `src.train` *does not* run preprocessing for you — step 6 must have
been run at least once, or training stops at a missing
`data/processed/train.csv`.

---

### 8. Make Predictions

Three ways to use the trained models. All of them require steps 6 and 7 to
have completed.

#### A. Batch Inference (`src.ensemble`)
Classifies an entire CSV or Excel file and writes a predictions file:
```
 python -m src.ensemble     --raw_csv data/processed/test.csv     --out predictions.csv
```
The output file is the input plus three columns per target — `pred_<target>`,
`confidence_<target>` and `review_<target>` (true when confidence is below
`review_threshold`, flagging rows worth a human look). Wherever the input carries a target's
source column — as `test.csv` does — accuracy and a per-category report are
printed for that target too, which makes the command above a quick way to check
how well training went.

`--raw_csv` here takes a **real path**, unlike in step 6. Model and label-map
locations come from `config.json`; override them with `--models_dir` and
`--data_dir` if they live elsewhere.

#### B. FastAPI Server
```
uvicorn src.api:app --reload
```
Interactive docs at http://localhost:8000/docs. One TIR per request:
```
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"description": "cracked weld on pipe hanger"}'
```
The response carries one entry per configured target:
```json
{
  "text": "cracked weld on pipe hanger",
  "predictions": {
    "process_cat": { "id": 18, "label": "WE WELD", "confidence": 0.86, "review_flag": false },
    "metric_cat":  { "id": 6,  "label": "WS Workmanship", "confidence": 0.99, "review_flag": false }
  }
}
```

#### C. Streamlit Web App
```
streamlit run src/streamlit_app.py
```
Opens at http://localhost:8501 with two tabs:

- **Single TIR** — paste the description and get a prediction for every
  configured target side by side, each with its own confidence figure and a
  review warning when confidence is below the threshold. If one is wrong, pick
  the right category from that field's dropdown and press *Save correction* —
  see **Learning from corrections** below. A correction is recorded against
  the field it was made on, so fixing one never implies anything about the
  other.
- **Batch file** — upload a CSV or Excel export, classify every row against
  every target, preview the result, and download it as `predictions.csv` with
  `pred_`, `confidence_` and `review_` columns per target.
- **Corrections** — every correction made so far, how many are in effect, and
  a download of `data/feedback.csv`.

Run it from the repository root. If the models have not been trained yet the
page says so and shows the commands to run instead of failing.

---

### Learning from Corrections

When a reviewer fixes a wrong category in the web app, the correction is
written to `data/feedback.csv` with a timestamp, the text, and the predicted
and corrected labels.

**By default, corrections are learned by retraining** (see *Permanently*
below). Nothing about a prediction changes until `src.train` runs, so a given
set of models always produces the same answer — the behaviour is reproducible
from the artifacts alone, with no hidden per-session state.

**Optionally, corrections can also apply immediately.** Set
`feedback.enabled` to `true` in `config/config.json` and a corrected TIR comes
back with the reviewer's category on the very next classification, labelled
*"Answered from a previous reviewer correction, not the model."* — no
retraining, no restart, in both the single and batch tabs. Useful for
demonstrations; it does mean the running app can answer differently from the
models on disk, which is why it is off by default.

With it enabled, matching is exact on the cleaned text, and otherwise by
cosine similarity of the dense embeddings, so a lightly reworded TIR still
benefits:

| Wording, versus the corrected text | Similarity | Reused? |
| --- | --- | --- |
| Identical | 1.00 | yes |
| One extra word | 0.92 | yes |
| One typo | 0.94 | yes |
| Genuinely reworded | 0.69 | no — left to the model |
| Unrelated TIR | 0.00 | no |

The 0.88 default was calibrated on 5.3 M different-category pairs from the
sample export, where the 99th percentile similarity is 0.23 and only 0.0003 %
of pairs reach 0.88 at all. Tune it in `config/config.json`:

```json
"feedback": { "enabled": false, "similarity_threshold": 0.88 }
```

Raise the threshold toward `1.0` to require closer matches (`1.0` means exact
text only). `enabled` controls only the immediate override — corrections are
logged, and used by the next retrain, either way.

**Permanently.** `src.train` folds every correction into the training split, so
the models themselves learn the mistake:

```
python -m src.train
```

The run reports how many were added. Corrections join the **training** split
only — validation and test are left untouched, so accuracy figures stay
honest. A correction naming a category that is not in that target's label map is
skipped (adding a genuinely new category means re-running `src.preprocess`).

When enabled, the override is deliberately limited to the web app.
`src.ensemble` is what you use to measure accuracy, and silently substituting
reviewer answers there would inflate the numbers it reports.

---

### 9. Optional: Verify the Install
```
python test/test_embedding.py
python test/test_feedback.py
```
Runs the embedding sanity checks (determinism, shapes, metadata round-trip)
and the correction-store checks (matching, thresholds, per-target filtering).
Both take a second and need no trained models — a quick check that the install
is sound. They locate the modules themselves, so they run from any directory.

> Run them as scripts, not with `python -m test.test_embedding`. The folder
> name `test` collides with CPython's own stdlib `test` package, which wins
> the import and makes the module form fail. Renaming the folder to `tests`
> would also resolve it.

`.gitlab-ci.yml` runs both, plus a compile pass and a config check, on every
push — so a merge request shows the result without anyone running them by
hand. **It must sit at the repository root**, the folder holding `.git`, which
in this repo is one level above `TIR_Production`; GitLab ignores copies placed
anywhere else. Its `PROJECT_DIR` variable holds that offset.

### 10. Optional: Tune the Ensemble Weights
```
python -m src.tune_weights            # report
python -m src.tune_weights --write    # apply to config.json
```
Grid-searches the XGBoost/SVM/LR blend against the **validation** split and
reports the best set. Tuning on test would make the reported accuracy
optimistic, so it never touches it. Run after training; no retrain is needed
afterwards, since the weights apply at inference.

---

## Troubleshooting

| Message | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root using `python -m src.<module>`, not `python src/<module>.py`. |
| `Missing optional dependency 'openpyxl'` | Reading `.xlsx` needs openpyxl; reinstall with `pip install -r requirements.txt`. |
| `FileNotFoundError: Raw data file not found: ...` | The file is not in `data/raw/`. Pass only the file name, not a path (step 6). |
| `ValueError: Input file is missing required column(s): ...` | The raw export lacks `Description 1` (or `Process Cat`, when preprocessing). |
| `FileNotFoundError: ... models/tfidf_word.pkl` | Models have not been trained yet — run step 7. |
| `FileNotFoundError: ... models/<target>/svm.pkl` | A target in `config.json` has no trained models — rerun steps 6 and 7. |
| `Model integrity check failed for ...` | An artifact changed since training. Retrain, or restore the original file. |
| `X has N features, but ... is expecting M features` | Models and settings are out of sync — usually the `embedding` block in `config/config.json` changed after training. Rerun steps 6 and 7. |

---

## Accuracy and the Review Threshold

Measured on 1,317 held-out rows, with the `Description 1` text only:

| Target | Categories | Accuracy | Always-guess baseline | Lift |
| --- | --- | --- | --- | --- |
| Process Cat | 22 | ~86 % | 29.3 % | **+57 points** |
| Metric Cat | 7 | ~98 % | 96.7 % | +1.9 points |

**Quote these as ±1 point.** Re-running the same code and data with a different
random split moves Process Cat accuracy between 85.2 % and 88.2 % — a 3-point
spread, standard deviation 0.0099 over six splits. Any single run's figure is
one draw from that range, so differences under a point mean nothing.

Read the baseline column alongside the accuracy. 96.7 % of records share one
Metric Cat, so guessing beats the deck's ~80 % bar on that field without any
model; the honest measure of what the model contributes is the lift. Process
Cat is the field where classification earns its keep.

### What the threshold buys

`review_threshold` in `config/config.json` sets the confidence below which a
row is flagged for a human. Because the model reports a confidence per row, it
can be set to trade coverage against accuracy:

| Threshold | Auto-approved | Accuracy on those | Sent to a coder |
| --- | --- | --- | --- |
| 0.55 | 92.0 % | 89.2 % | 8.0 % |
| 0.70 | 84.6 % | 91.7 % | 15.4 % |
| 0.80 | 79.0 % | 93.2 % | 21.0 % |
| **0.90** (default) | **62.5 %** | **95.4 %** | **37.5 %** |
| 0.95 | 49.2 % | 97.8 % | 50.8 % |

At the 0.90 default, roughly six in ten TIRs are coded at about 95 % accuracy
and the remaining four are routed to a coder — which is the number to use when
estimating time saved, rather than the headline accuracy. Lower the threshold
for more coverage, raise it for more certainty.

Metric Cat is far less sensitive: it stays near 98 % accuracy at 98.8 %
coverage even at 0.90, because one category dominates.

---

## Reproducibility
The system is fully deterministic. Given identical raw data, preprocessing, embeddings, training, and predictions will always be the same. All model artifacts include a SHA‑256 digest, which is verified *before* the artifact is loaded.


### Feedback

if classication is wrong have a drop down menu showing possible categories and end user can select the right anwser. 