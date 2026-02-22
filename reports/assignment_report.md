# PySpark Data Engineering, Distributed ML, and Tableau Assignment

Dataset used: UCI Online Retail
- Dataset page: `https://archive.ics.uci.edu/dataset/352/online+retail`
- Download URL: `https://archive.ics.uci.edu/static/public/352/online+retail.zip`

## 1. Pyspark Data Engineering

### 1a) Data Ingestion and Storage Design
- SparkSession optimization configured in `src/spark_utils.py` (AQE, skew handling, shuffle tuning, Kryo, Parquet Snappy).
- Ingestion in `src/01_ingest.py` reads UCI dataset from URL or local `.zip/.xlsx`.
- Validation at ingestion:
  - non-empty `invoice_no`, `stock_code`, `country`
  - `quantity != 0`
  - `unit_price > 0`
  - valid `customer_id`
- Invalid rows go to quarantine: `outputs/quarantine/online_retail`.
- Bronze stored as partitioned Parquet by `country`, `invoice_year`, `invoice_month`.

### 1b) Distributed Data Processing Pipeline
- `src/02_pipeline.py` performs cleaning, dedupe, and enrichment with a small country dimension.
- Broadcast join enabled via `--use_broadcast`; non-broadcast path also implemented for comparison.
- Memory strategy uses `persist(MEMORY_AND_DISK)` and explicit `unpersist()`.
- Error handling + lineage logging implemented in all pipeline scripts using `outputs/logs/lineage.jsonl`.

### 1c) Performance Optimization
- DataFrame API used throughout (Catalyst + Tungsten optimizations).
- Caching at repeated-computation points in pipeline and scaling scripts.
- Shuffle/partition tuning via app-specific `spark.sql.shuffle.partitions` and `repartition` strategy.
- Broadcast vs non-broadcast evidence produced by:
  - `src/02_compare_broadcast.py`
  - `outputs/tableau/broadcast_compare_csv/summary.csv`
  - plan dumps in `artifacts/spark_ui/*.txt`

## 2. Scalability and Distributed ML

### 2a) PySpark MLlib Implementation
- 3 MLlib algorithms in `src/03_ml.py`:
  - LogisticRegression
  - RandomForestClassifier
  - GBTClassifier
- sklearn baseline (single-node): logistic regression + random forest.
- Custom transformer: `BasketSignalTransformer` for retail-specific features.
- Model serialization:
  - MLlib models saved under `outputs/models/mllib_*`
  - sklearn models saved with pickle: `outputs/models/sklearn_models.pkl`

### 2b) Distributed Training & Hyperparameter Tuning
- CrossValidator with `parallelism=4`, `numFolds=3`.
- RF grid constrained to 12 combinations for practical runtime.
- Checkpointing enabled via `sparkContext.setCheckpointDir` and dataframe checkpointing.
- Resource profile justification documented in scaling output and report.

### 2c) Scalability Analysis
- Strong and weak scaling implemented in `src/04_scaling.py`.
- Metrics exported:
  - runtime
  - throughput
  - estimated cost
  - cost per 1000 rows
  - strong efficiency
- Bottleneck analysis supported via Spark UI (I/O, shuffle/network, compute stages).

## 3. Tableau Visualization

### 3a) Dashboard Plan
- Dashboard 1: Data quality + pipeline monitoring (`dq_metrics_csv`, lineage logs).
- Dashboard 2: Model performance (`model_metrics_csv`).
- Dashboard 3: Business insights (`sales_metrics_csv`, `evaluation_csv`).
- Dashboard 4: Scalability + cost (`scaling_csv`, `broadcast_compare_csv`).

### 3b) Best Practices
- Use Tableau extracts for speed.
- Use LOD expressions for country/region-level analysis.
- Use parameters for model/resource profile exploration.
- Create mobile layouts with KPI-first cards.

### 3c) Storytelling
- Flow: Data trust -> Model quality -> Business value -> Operational cost.
- Annotate key threshold shifts (AUC uplift, cost/throughput breakpoints).
- Use action filters between country/region and model/scaling dashboards.

## 4. Model Evaluation

### 4a) Distributed Evaluation Metrics
- Temporal split using `invoice_ts` (train/validation/test).
- Stratified sampling for class balance checks.
- Bootstrap confidence intervals for AUC in `src/05_evaluate.py`.
- Business metric alignment via expected campaign profit function.

## Reproducible Execution

```bash
python3 src/01_ingest.py --max_records 0
python3 src/02_compare_broadcast.py --bronze_path outputs/bronze/online_retail
python3 src/02_pipeline.py --bronze_path outputs/bronze/online_retail --gold_path outputs/gold --use_broadcast
python3 src/03_ml.py
python3 src/04_scaling.py
python3 src/05_evaluate.py
```
