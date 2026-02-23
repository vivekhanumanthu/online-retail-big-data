# online-retail-assignment

PySpark + MLlib + Tableau assignment implementation using the UCI Online Retail dataset:
`https://archive.ics.uci.edu/dataset/352/online+retail`

## Dataset Source
Default ingestion URL used by `src/01_ingest.py`:
`https://archive.ics.uci.edu/static/public/352/online+retail.zip`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Run Commands

```bash
python3 src/01_ingest.py --max_records 0
python3 src/02_compare_broadcast.py --bronze_path outputs/bronze/online_retail
python3 src/02_pipeline.py --bronze_path outputs/bronze/online_retail --gold_path outputs/gold --use_broadcast
python3 src/03_ml.py --gold_path outputs/gold --model_path outputs/models
python3 src/04_scaling.py --gold_path outputs/gold/model_features --output_path outputs/tableau/scaling_csv
python3 src/05_evaluate.py --features_path outputs/gold/model_features --model_path outputs/models/mllib_rf_tuned --output_path outputs/tableau/evaluation_csv
```

## Relevant Outputs
- Bronze: `outputs/bronze/online_retail`
- Quarantine: `outputs/quarantine/online_retail`
- Gold: `outputs/gold/*`
- Models: `outputs/models/*`
- Tableau CSVs: `outputs/tableau/*`
- Broadcast comparison: `outputs/tableau/broadcast_compare_csv/summary.csv`
- Spark plan evidence: `artifacts/spark_ui/02_pipeline_plan_broadcast.txt`, `artifacts/spark_ui/02_pipeline_plan_no_broadcast.txt`
- Reports: `reports/assignment_report.md`, `reports/insights_report.md`, `reports/rubric_checklist.md`
