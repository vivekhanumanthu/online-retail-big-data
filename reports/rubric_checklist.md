# Rubric Checklist

## 1. Pyspark Data Engineering
- [x] SparkSession config optimization (`src/spark_utils.py`)
- [x] Partitioning strategy (`src/01_ingest.py`: country/year/month)
- [x] Parquet storage with justification (`reports/assignment_report.md`)
- [x] Ingestion validation + quarantine (`src/01_ingest.py`)
- [x] Broadcast join (`src/02_pipeline.py`)
- [x] persist/unpersist strategy (`src/02_pipeline.py`)
- [x] Error handling + lineage (`src/*.py`, `outputs/logs/lineage.jsonl`)
- [x] DataFrame vs RDD justification (`reports/assignment_report.md`)
- [x] Caching strategy documentation (`reports/assignment_report.md`)
- [x] Spark UI evidence path (`artifacts/spark_ui/README.md`)
- [x] Shuffle/partition tuning (`src/*.py`)

## 2. Scalability and Distributed ML
- [x] 3 MLlib algorithms (`src/03_ml.py`)
- [x] sklearn baseline comparison (`src/03_ml.py`)
- [x] Custom transformer (`BasketSignalTransformer`)
- [x] MLlib + pickle serialization (`src/03_ml.py`)
- [x] CrossValidator + parallelism (`src/03_ml.py`)
- [x] Constrained hyperparameter grid (`src/03_ml.py`)
- [x] Checkpointing (`src/03_ml.py`)
- [x] Resource allocation/cost-performance analysis (`src/04_scaling.py`)
- [x] Strong scaling (`src/04_scaling.py`)
- [x] Weak scaling (`src/04_scaling.py`)
- [x] Bottleneck analysis support (`reports/assignment_report.md`)

## 3. Tableau Visualization
- [x] Dashboard strategy for data quality
- [x] Dashboard strategy for model performance
- [x] Dashboard strategy for business insights
- [x] Dashboard strategy for scalability/cost
- [x] Extract strategy, LOD, parameters, mobile, storytelling
(All documented in `reports/assignment_report.md`)

## 4. Model Evaluation
- [x] Temporal split (`src/03_ml.py`, `src/05_evaluate.py`)
- [x] Stratified sampling (`src/05_evaluate.py`)
- [x] Bootstrap confidence intervals (`src/05_evaluate.py`)
- [x] Business metric alignment (`src/05_evaluate.py`)
