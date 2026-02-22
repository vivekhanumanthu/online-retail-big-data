#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end runner for the assignment
# Uses UCI Online Retail dataset.

python3 src/01_ingest.py \
  --max_records 0

python3 src/02_pipeline.py
python3 src/03_ml.py
python3 src/04_scaling.py
python3 src/05_evaluate.py

echo "All assignment pipelines completed successfully."
