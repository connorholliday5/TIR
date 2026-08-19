
# TIR Liability Coding System

The TIR Liability Coding System reads the description text on a Technical Issue
Report and suggests the QPS liability codes for it. It is built to assist the
coding team rather than replace them: every answer carries a confidence, and
anything the system is unsure of is flagged for a person rather than guessed at
quietly.

The whole pipeline is automated. Point it at one or more QPS exports and it
cleans the text, builds the training data, fits the models, and produces both a
web app for coders and a command-line tool for whole files.

---

## How the System Works (Non-Technical Overview)

### 1. Raw Data

Put the QPS exports anywhere convenient — beside `TIR_Production/`, inside
`TIR_Production/data/raw/`, or give a full path. All three are found
automatically.

**Any QPS export layout is accepted.** The same field is named differently
depending on which report produced the file, and all of these are understood as
the same column:

| The field | Named this in one export | And this in another | And this in a third |
| --- | --- | --- | --- |
| Description 1 | `Description 1` | `DESCRIPTION_ONE` | `Item Description 1` |
| Process Cat | `Process Cat` | `PROCESS_CATEGORY` | `Item Process Category` |
| Metric Cat | `Metric Cat` | `METRIC_CATEGORY` | `Item Metric Category` |

The full list lives in the `column_aliases` block of `config/config.json`, so a
fourth export layout is a settings change rather than a code change.

### 2. Preprocessing

`src/preprocess.py` renames every recognised column to a single canonical name,
then:

- Joins Description 1, Description 2 and the Doc Title into the text the models
  read
- Removes records that appear in more than one export
- Standardises the category names and assigns each a numeric label
- Records which sub-codes belong under which category
- Splits the result into training, validation and test sets

On the two current exports that means **103,281 rows in, 12,321 duplicates
removed, 90,960 kept** — split into 65,718 for training, 11,598 for checking the
work during training, and 13,644 held back and never looked at until the end.

> The smaller export turns out to be 99.4 % contained in the larger one. Because
> the two name their columns differently, a plain row-by-row comparison finds
> nothing in common, which is why de-duplication happens *after* the columns are
> made consistent.

### 3. What It Codes

Four fields, in order, because the last two are chosen inside the first two:

| Field | Categories | Records available to learn from |
| --- | --- | --- |
| Metric Cat | 7 | 90,958 (100 %) |
| Process Cat | 27 | 61,263 (67 %) |
| Process Sub | 144 | 60,692 (67 %) |
| Process Level 3 | 543 | 55,069 (60 %) |

Process Cat is blank on a third of records because informational items are
rarely coded. Those rows are skipped for that field and still used for the
fields they do carry.

### 4. Codes That Fit Together

Process Sub is chosen from the sub-codes that belong under the predicted
Process Cat, and Process Level 3 from those under the predicted Sub. A
combination QPS would reject cannot be produced.

This also makes the deepest level workable at all: one model choosing between
543 codes is neither accurate nor practical, while roughly 85 small models —
one per parent, each choosing between a handful — is both.

### 5. Model Training

Text is turned into features two ways, both from the description itself:

- TF-IDF word features (1–2 grams)
- TF-IDF character features (3–5 grams)

For each field, a **calibrated linear support vector machine** is fitted, and
logistic regression, stochastic gradient and gradient boosting are fitted
alongside it. Any that scores better is kept and blended in; the rest are
discarded. The comparison is recorded in `models/<field>/ensemble.json` so the
choice can be checked rather than taken on trust.

Calibration is what makes the confidence a real probability — see
**Accuracy and the Review Threshold** below for why that matters.

### 6. Prediction Tools

- **Streamlit web app** — code one TIR at a time, collect a session's worth, and
  export them all as one spreadsheet
- **`src.classify`** — code a whole file from the command line
- **`src.api`** — an HTTP endpoint for another system to call

---

## Project Structure

The repository root holds the source exports and briefing material; the project
itself lives in `TIR_Production/`.

For what each file does and how it works, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
.gitignore
README.md
Original - QPS TIR and SURV Import 080326.xlsx   The QPS write-back template
QPS Pull Jan 2023 - Dec 2025 ....xlsx            Main training export
TIR Export Example1.xlsx                         Smaller export, largely a subset
SY4.0_AIML_TIR_TechNav_aRosenbloom.pdf           Project quad chart
ARCHITECTURE.md                                  What each file does and how it works

TIR_Production/
  config/config.json           Targets, column aliases, thresholds, model comparison
  data/raw/                    Raw TIR files (optional; not tracked)
  data/processed/              Cleaned splits, written by src.preprocess
  data/label_map_*.json        One per field: label ID -> category name
  data/hierarchy.json          Which sub-codes belong under which category
  data/feedback.csv            Coder corrections logged from the web app
  logs/
  models/                      Shared encoders, plus one folder per field
  reports/                     The two study reports, regenerable
  requirements.txt
  pyrightconfig.json           Type-checker settings, so an editor agrees with CI

  src/config.py                Paths and settings, read once for everything else
  src/data.py                  Recognising an export's columns; building the text
  src/models.py                Features, blending, artifact integrity
  src/preprocess.py            Cleaning, labelling, splitting, hierarchy
  src/train.py                 Training and the model comparison
  src/inference.py             Loading the models and coding a TIR
  src/feedback.py              Coder corrections: stored, reused, retrained on
  src/export.py                The session spreadsheet
  src/classify.py              Batch inference from the command line
  src/reports.py               The data-sufficiency and benchmark reports
  src/app.py                   Streamlit web app
  src/api.py                   HTTP endpoint

  test/test_data.py            Column recognition across every export layout
  test/test_models.py          Text assembly, blending, correction reuse
  test/test_reports.py         The agreement measure and the coverage search
  train.log                    Full technical detail of the last training run
```

Everything under `models/`, `data/processed/` and the label maps is generated —
`src.preprocess` and `src.train` rebuild all of it, so none of it is tracked.

---

## Startup Guide

Steps 1–5 take you from a fresh clone to working predictions. Run them in
order, from `TIR_Production/`, in one terminal session.

### 1. Clone the Repository

```bash
git clone https://github.com/connorholliday5/TIR.git
cd TIR
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
cd TIR_Production
pip install -r requirements.txt
```

Check it before spending ten minutes on a build:

```bash
python -m pytest test
```

46 tests, about two seconds. They confirm the wiring without needing any models.

### 4. Build the Training Data

```bash
python -m src.preprocess --raw_csv "QPS Pull Jan 2023 - Dec 2025 Non Nuc_Norforn-N_TIR_SURV_Validated.xlsx" "TIR Export Example1.xlsx"
```

About three minutes. It reports what it recognised in each file, how many
duplicates it removed, and how many records each field can be learned from.

If it cannot find a file it lists every folder it looked in.

### 5. Train the Models

```bash
python -m src.train
```

The output names each field as it goes and finishes with a summary of all four.

**This compares four model families and takes hours.** While iterating, skip the
comparison and fit the linear model alone:

```bash
python -m src.train --no-gate
```

That takes about seven minutes and is what you want unless you are specifically
asking which model family wins.

### 6. Code Some TIRs

#### A. The Web App

```bash
streamlit run src/app.py
```

Type a description, add any identifying details you have, and read the four
codes. **Confirm or correct the Process Cat before the deeper codes are
chosen** — see the note under the accuracy table for why this matters more than
anything else on the page. Everything you code in a sitting is collected, and
**Download Excel** saves the lot as one spreadsheet.

#### B. A Whole File

```bash
python -m src.classify --raw_csv "<export>.xlsx" --out predictions.csv
```

Any of the export layouts works. If the file carries its own codes, accuracy is
reported against them.

#### C. HTTP Endpoint

```bash
uvicorn src.api:app --reload
```

`POST /predict` with a description and it returns the four codes with their
confidences.

### 7. Optional: Regenerate the Reports

```bash
python -m src.reports all --raw_csv "QPS Pull Jan 2023 - Dec 2025 Non Nuc_Norforn-N_TIR_SURV_Validated.xlsx" "TIR Export Example1.xlsx"
```

Writes `reports/data_sufficiency.md` (is the data good enough to learn from?)
and `reports/benchmark.md` (what are the models worth in coder time?).

---

## Learning from Corrections

Correcting a code in the web app writes it to `data/feedback.csv` and it is used
twice:

1. **Immediately.** A TIR whose wording matches one already corrected gets the
   correction rather than the model's answer. Matching is exact, or above 0.80
   similarity for near-duplicates — calibrated so a typo (0.815) or an inserted
   word (0.842) still matches while a genuine rewording (0.207) does not.
2. **Permanently.** `src.train` folds every correction into the training data on
   the next run.

---

## Accuracy and the Review Threshold

Measured on the 13,644 held-out records:

| Field | Gets it right | Macro-F1 | Coders agree with each other |
| --- | --- | --- | --- |
| Metric Cat | 92.8 % | 0.605 | 91.4 % |
| Process Cat | 88.2 % | 0.743 | 85.3 % |
| Process Sub | 80.8 % | 0.630 | 75.4 % |
| Process Level 3 | 73.6 % | 0.540 | 67.0 % |

**Read the last column with the first.** Where the same description was coded
more than once in the history, the codes given differ that often. A model cannot
be more consistent than the data it learns from, so every field is already at or
slightly past its ceiling. A better model does not move these numbers; better-
agreed codes do.

Macro-F1 counts every category equally, so a category coded daily and one coded
twice a year weigh the same. It is much harsher than "percent correct" and it is
the number that shows how the rare categories are doing.

### What the threshold buys

`review_threshold` is set per field in `config/config.json`. Holding the
automatic answers to **95 % precision**:

| Field | Coded automatically | Left for a coder | Threshold that reaches 95 % | Currently set to |
| --- | --- | --- | --- | --- |
| Metric Cat | 94.3 % | 5.7 % | 0.63 | 0.80 |
| Process Cat | 80.3 % | 19.7 % | 0.69 | 0.69 |
| Process Sub | not reachable | — | — | 0.60 |
| Process Level 3 | not reachable | — | — | 0.50 |

Metric Cat is set deliberately higher than it needs to be: at 0.80 it codes
86.1 % of records at 97.3 % precision rather than 94.3 % at 95 %. Lower it for
more coverage.

Process Cat coding four TIRs in five, correct nineteen times in twenty, is the
figure to use when estimating time saved.

The two deeper fields top out near 92 % and 94 % precision even when only the
most confident answers are kept, so 95 % is not available at any setting. At a
90 % standard they cover 70.6 % and 50.8 %. They can assist a coder; they should
not code unsupervised.

### Confirming the category is worth more than any model change

Process Sub is chosen by the model belonging to its parent. Routed by a
*predicted* Process Cat it scores 80.9 %; routed by a *confirmed* one, 90.3 %.

**That one click is worth 9.4 points** — more than every feature and
hyper-parameter change tested put together. It is why the web app asks for the
category first.

---

## Reproducibility

Given the same raw data, preprocessing and training produce the same result
every time. Every model artifact carries a SHA-256 digest that is verified
*before* the artifact is loaded, since loading one executes code.

---

## Feedback, Upgrades, and What an Upgrade Needs

### Feedback that has been acted on

| Asked for | Where it went |
| --- | --- |
| If a classification is wrong, offer a drop-down of possible categories | On the Single TIR tab. Correcting the Process Cat also re-chooses the codes beneath it. |
| Read the other QPS exports, not just the one | Any of the three layouts is recognised automatically. |
| Code the sub-category and third level, not only the top two | Both added, each chosen inside the level above it. |
| Export the single TIRs done in a session | Collected as you go; one button saves them all as a spreadsheet. |

### What would raise the numbers, in order of what it is worth

Accuracy is capped by how consistently the source data was coded, so the largest
gains are not modelling changes.

1. **Adjudicate the conflicting codes.** Where the same description was coded
   twice, 31 % of Process Cat pairs and 52 % of Level 3 pairs disagree. Settling
   those is the only thing that raises the ceiling itself.
2. **Write a tie-breaker rule for the ambiguous pairs.** A model cannot learn a
   rule the coders do not share. One page of "when it is both X and Y, code X"
   turns judgement into signal.
3. **Retire or merge the dead Level 3 codes.** 536 of them have fewer than ten
   records ever. They cannot be learned and mostly create disagreement.
4. **Make Description 2 mandatory in QPS.** It is filled on 28 % of records, and
   adding it gained 5.4 points of macro-F1 — real signal that is usually absent.
5. **Code the informational items, or exclude them by rule.** A third of records
   carry no Process Cat, so the model never learns from them.

### What an upgrade would need from outside the code

- **Double-code 200 TIRs** to measure the true ceiling. The agreement figures
  here are a lower bound taken from records sharing a Description 1, and two
  such records can legitimately be different events. A deliberate double-coding
  exercise says what "100 %" actually means for this data before anyone targets
  it.
- **A decision on the deeper levels.** Process Sub and Level 3 cannot reach 95 %
  precision. Either they assist rather than auto-code, or the taxonomy is
  simplified. That is a process call, not a technical one.
- **If a language model is wanted:** it needs sign-off for TIR text to leave the
  machine, or an internal endpoint. It would help the rare-category tail, where
  there are too few examples for the current approach — it would **not** raise
  the ceiling, which is set by label consistency.
