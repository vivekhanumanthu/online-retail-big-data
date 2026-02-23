#!/usr/bin/env bash
set -euo pipefail

python3 src/01_ingest.py --max_records 0
python3 src/02_compare_broadcast.py --bronze_path outputs/bronze/online_retail
python3 src/02_pipeline.py --bronze_path outputs/bronze/online_retail --gold_path outputs/gold --use_broadcast
python3 src/03_ml.py --gold_path outputs/gold --model_path outputs/models
python3 src/04_scaling.py --gold_path outputs/gold/model_features --output_path outputs/tableau/scaling_csv
python3 src/05_evaluate.py --features_path outputs/gold/model_features --model_path outputs/models/mllib_rf_tuned --output_path outputs/tableau/evaluation_csv

echo "All assignment pipelines completed successfully."
