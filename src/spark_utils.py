from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict

from pyspark.sql import SparkSession


def build_spark(app_name: str, shuffle_partitions: int = 400) -> SparkSession:
    """Create a SparkSession tuned for medium-size distributed workloads."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", "134217728")
        .config("spark.sql.broadcastTimeout", "1200")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def log_lineage(
    pipeline_name: str,
    input_path: str,
    output_path: str,
    row_count: int,
    extra: Dict[str, str] | None = None,
    log_path: str = "outputs/logs/lineage.jsonl",
) -> None:
    """Append a lineage event for auditability and debugging."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    event = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "pipeline_name": pipeline_name,
        "input_path": input_path,
        "output_path": output_path,
        "row_count": row_count,
        "extra": extra or {},
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def safe_stop(spark: SparkSession) -> None:
    try:
        spark.stop()
    except Exception:
        pass
