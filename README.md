# online-retail-assignment

PySpark + MLlib + Tableau assignment implementation using the UCI Online Retail dataset:
`https://archive.ics.uci.edu/dataset/352/online+retail`

## Dataset Source
Default ingestion URL used by `src/01_ingest.py`:
`https://archive.ics.uci.edu/static/public/352/online+retail.zip`

## Naming Convention
- `src/01_ingest.py`
- `src/02_pipeline.py`
- `src/02_compare_broadcast.py`
- `src/03_ml.py`
- `src/04_scaling.py`
- `src/05_evaluate.py`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
./run_all.sh
```

Recommended Python version: `3.10` or `3.11`.

## Ingestion Options
Use UCI URL directly (default):

```bash
python3 src/01_ingest.py --max_records 0
```

Use local source file (`.zip` or `.xlsx`):

```bash
python3 src/01_ingest.py --input_path data/raw/online_retail.zip --max_records 100000
```

## Broadcast Join Comparison
Run both versions and export comparison evidence:

```bash
python3 src/02_compare_broadcast.py --bronze_path outputs/bronze/online_retail
```

Outputs:
- `outputs/tableau/broadcast_compare_csv/summary.csv`
- `artifacts/spark_ui/02_pipeline_plan_broadcast.txt`
- `artifacts/spark_ui/02_pipeline_plan_no_broadcast.txt`

## Main Outputs
- Bronze data: `outputs/bronze/online_retail`
- Quarantine: `outputs/quarantine/online_retail`
- Gold data: `outputs/gold/*`
- Models: `outputs/models/*`
- Tableau CSV extracts: `outputs/tableau/*`
- Lineage log: `outputs/logs/lineage.jsonl`

## Reports
- Full rubric report: `reports/assignment_report.md`
- Insight report: `reports/insights_report.md`
- Rubric checklist: `reports/rubric_checklist.md`
