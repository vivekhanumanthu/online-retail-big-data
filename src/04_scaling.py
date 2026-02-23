from __future__ import annotations

import argparse
import time
from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_utils import build_spark, log_lineage, safe_stop


def workload(df: DataFrame, n_partitions: int) -> int:
    q = (
        df.repartition(n_partitions, "region")
        .groupBy("region", "country")
        .agg(
            F.count("*").alias("invoice_rows"),
            F.avg("invoice_total").alias("avg_invoice_total"),
            F.avg("total_units").alias("avg_total_units"),
            F.avg("shipping_risk_score").alias("risk"),
        )
        .orderBy(F.desc("invoice_rows"))
    )
    return q.count()


def run_case(df: DataFrame, mode: str, resources: str, sample_fraction: float, n_partitions: int) -> Tuple[str, str, float, int, float]:
    test_df = df.sample(False, sample_fraction, 42) if sample_fraction < 1.0 else df
    start = time.perf_counter()
    result_count = workload(test_df, n_partitions)
    sec = time.perf_counter() - start

    e = int(resources.split("_")[0].replace("E", ""))
    c = int(resources.split("_")[1].replace("C", ""))
    m = int(resources.split("_")[2].replace("M", ""))
    est_cost = (0.03 * e * c + 0.004 * e * m) * (sec / 3600)

    return mode, resources, sec, result_count, est_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strong/weak scaling experiments")
    parser.add_argument("--gold_path", type=str, default="outputs/gold/model_features")
    parser.add_argument("--output_path", type=str, default="outputs/tableau/scaling_csv")
    args = parser.parse_args()

    spark = build_spark("online-retail-scaling", shuffle_partitions=500)

    try:
        df = spark.read.parquet(args.gold_path).cache()
        df.count()

        strong_cases: List[Tuple[str, str, float, int, float]] = []
        weak_cases: List[Tuple[str, str, float, int, float]] = []

        for resources, n_partitions in [("E4_C4_M16", 120), ("E8_C4_M16", 240), ("E12_C4_M16", 360)]:
            strong_cases.append(run_case(df, "strong", resources, sample_fraction=0.6, n_partitions=n_partitions))

        for resources, frac, n_partitions in [("E4_C4_M16", 0.2, 120), ("E8_C4_M16", 0.4, 240), ("E12_C4_M16", 0.6, 360)]:
            weak_cases.append(run_case(df, "weak", resources, sample_fraction=frac, n_partitions=n_partitions))

        rows = strong_cases + weak_cases
        out_df = spark.createDataFrame(rows, ["mode", "resource_profile", "runtime_sec", "result_rows", "estimated_cost_usd"])

        baseline_strong = min([r[2] for r in strong_cases]) if strong_cases else 1.0
        out_df = (
            out_df.withColumn("throughput_rows_per_sec", F.col("result_rows") / F.greatest(F.col("runtime_sec"), F.lit(0.001)))
            .withColumn(
                "cost_per_1000_rows",
                (F.col("estimated_cost_usd") / F.greatest(F.col("result_rows"), F.lit(1))) * F.lit(1000.0),
            )
            .withColumn(
                "strong_efficiency",
                F.when(F.col("mode") == "strong", F.lit(baseline_strong) / F.col("runtime_sec")).otherwise(F.lit(None)),
            )
        )

        out_df.coalesce(1).write.mode("overwrite").option("header", True).csv(args.output_path)

        log_lineage(
            pipeline_name="04_scaling",
            input_path=args.gold_path,
            output_path=args.output_path,
            row_count=out_df.count(),
            extra={"dataset": "UCI Online Retail", "strong_cases": str(len(strong_cases)), "weak_cases": str(len(weak_cases))},
        )

        print("Scaling analysis complete")

    except Exception as e:
        log_lineage(
            pipeline_name="04_scaling_failed",
            input_path=args.gold_path,
            output_path=args.output_path,
            row_count=0,
            extra={"error": str(e)},
        )
        raise
    finally:
        safe_stop(spark)


if __name__ == "__main__":
    main()
