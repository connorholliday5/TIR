# TIR Liability Coding

Assists the coding of Technical Issue Report liabilities by predicting the
QPS category fields from a TIR's description.

## What it predicts

| Field | Categories | What it is |
| --- | ---: | --- |
| Metric Cat | 7 | The kind of quality issue |
| Process Cat | 27 | The discipline it belongs to |
| Process Sub | 144 | The sub-discipline, predicted inside the category above |
| Process Level3 | 543 | The specific finding, predicted inside the sub-discipline above |

The two deeper levels are predicted *within* whichever category the level
above them landed on, so the combination is always one QPS accepts.

## Reading any QPS export

The same field is named differently by each report the TIR team can run —
`Description 1`, `DESCRIPTION_ONE` and `Item Description 1` are all the same
column. Every layout is recognised automatically; the alias table lives in
`config.json` under `column_aliases`, so a new export format is a
configuration change rather than a code change.

## Running it

```bash
pip install -r requirements.txt

# Build the training data from one or more labelled exports
python -m src.preprocess --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"

# Fit the models
python -m src.train

# The two study deliverables
python -m src.report_data --raw_csv "QPS Pull ….xlsx" "TIR Export Example1.xlsx"
python -m src.report_benchmark

# Classify a file from the command line
python -m src.ensemble --raw_csv "<export>.xlsx" --out predictions.csv

# Or use the web app
streamlit run src/streamlit_app.py
```

The modules also run from a flat checkout, where every file sits at the
repository root: use `python preprocess.py …` and `streamlit run
streamlit_app.py` instead.

## Confirming the category is worth more than any model change

The sub-category is predicted by whichever classifier belongs to its parent.
Routed by a *predicted* Process Cat it scores about 81%; routed by a
*confirmed* one, about 90%. The single-TIR tab therefore asks a coder to
accept or correct the Process Cat first, and re-predicts the levels below it
from the answer. Every feature and hyper-parameter change tried was worth less
than that one click.

## What limits accuracy

Not the model. Where the same description appears more than once in the
history, coders gave it conflicting Process Cats about a quarter of the time,
and conflicting third-level codes about half the time. A classifier cannot be
more consistent than the data it learns from, so `reports/data_sufficiency.md`
reports that ceiling next to every accuracy figure, and
`reports/benchmark.md` reports what is actually useful: how much of a file can
be coded automatically at a chosen precision.
