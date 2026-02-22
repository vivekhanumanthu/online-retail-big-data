from __future__ import annotations

import argparse
import os
import time

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F

from spark_utils import build_spark, log_lineage, safe_stop


def build_country_dim(spark) -> DataFrame:
    rows = [
        ("United Kingdom", "europe", 1),
        ("Germany", "europe", 1),
        ("France", "europe", 1),
        ("Netherlands", "europe", 1),
        ("EIRE", "europe", 1),
        ("Australia", "apac", 2),
        ("Japan", "apac", 2),
        ("USA", "americas", 2),
        ("Canada", "americas", 2),
        ("unknown", "unknown", 3),
    ]
    return spark.createDataFrame(rows, ["country", "region", "shipping_risk_score"])


def clean_and_enrich(bronze_df: DataFrame, spark, use_broadcast: bool = True) -> DataFrame:
    country_dim = build_country_dim(spark)

    cleaned = (
        bronze_df.select(
            "transaction_id",
            "invoice_no",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "unit_price",
            "customer_id",
            "country",
            "line_total",
            "is_return",
            "invoice_ts",
            "invoice_year",
            "invoice_month",
            "ingested_at_utc",
        )
        .dropDuplicates(["transaction_id"])
        .withColumn("country", F.when(F.length(F.trim("country")) == 0, F.lit("unknown")).otherwise(F.col("country")))
    )

    if use_broadcast:
        enriched = cleaned.join(F.broadcast(country_dim), on="country", how="left")
    else:
        enriched = cleaned.join(country_dim, on="country", how="left")

    return enriched.fillna({"region": "unknown", "shipping_risk_score": 3})


def build_invoice_features(enriched: DataFrame) -> DataFrame:
    invoice = (
        enriched.groupBy("invoice_no", "customer_id", "country", "region", "shipping_risk_score", "invoice_ts")
        .agg(
            F.sum("line_total").alias("invoice_total"),
            F.sum(F.abs(F.col("quantity"))).alias("total_units"),
            F.countDistinct("stock_code").alias("unique_items"),
            F.avg("unit_price").alias("avg_unit_price"),
            F.max("is_return").alias("has_return_line"),
        )
        .withColumn("is_weekend", F.when(F.dayofweek("invoice_ts").isin([1, 7]), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("invoice_hour", F.hour("invoice_ts"))
        .withColumn("invoice_year", F.year("invoice_ts"))
        .withColumn("invoice_month", F.month("invoice_ts"))
    )

    w = Window.partitionBy("customer_id").orderBy("invoice_ts")
    invoice = (
        invoice.withColumn("next_invoice_ts", F.lead("invoice_ts").over(w))
        .withColumn("days_to_next_order", F.datediff("next_invoice_ts", "invoice_ts"))
        .withColumn("target_repeat_30d", F.when((F.col("days_to_next_order") >= 0) & (F.col("days_to_next_order") <= 30), 1).otherwise(0))
        .drop("next_invoice_ts")
    )

    return invoice


def build_metrics(enriched: DataFrame, invoice_features: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    sales_metrics = (
        enriched.groupBy("country", "region")
        .agg(
            F.countDistinct("invoice_no").alias("invoice_count"),
            F.countDistinct("customer_id").alias("customer_count"),
            F.sum("line_total").alias("gross_revenue"),
            F.avg("line_total").alias("avg_line_total"),
            F.sum("is_return").alias("return_line_count"),
        )
        .withColumn("return_line_ratio", F.col("return_line_count") / F.greatest(F.col("invoice_count"), F.lit(1)))
    )

    dq_metrics = (
        enriched.groupBy("country")
        .agg(
            F.count("*").alias("rows_after_cleaning"),
            F.approx_count_distinct("transaction_id").alias("unique_transactions"),
            F.avg("unit_price").alias("avg_unit_price"),
            F.avg("quantity").alias("avg_quantity"),
        )
        .withColumn("duplicate_ratio", 1 - (F.col("unique_transactions") / F.col("rows_after_cleaning")))
    )

    return sales_metrics, dq_metrics, invoice_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform bronze data to curated gold tables")
    parser.add_argument("--bronze_path", type=str, default="outputs/bronze/online_retail")
    parser.add_argument("--gold_path", type=str, default="outputs/gold")
    parser.add_argument("--use_broadcast", action="store_true", help="Use broadcast join for country dimension")
    parser.add_argument("--save_plan", action="store_true", help="Save physical plan text for evidence")
    parser.add_argument(
        "--plan_output",
        type=str,
        default="artifacts/spark_ui/02_pipeline_plan.txt",
        help="Path to save physical plan text when --save_plan is used",
    )
    args = parser.parse_args()

    join_mode = "broadcast" if args.use_broadcast else "no_broadcast"
    spark = build_spark(f"online-retail-pipeline-{join_mode}", shuffle_partitions=300)

    try:
        run_start = time.perf_counter()
        bronze_df = spark.read.parquet(args.bronze_path).repartition(200, "country")
        bronze_df = bronze_df.persist(StorageLevel.MEMORY_AND_DISK)

        enriched_df = clean_and_enrich(bronze_df, spark, use_broadcast=args.use_broadcast).persist(StorageLevel.MEMORY_AND_DISK)
        invoice_features = build_invoice_features(enriched_df).persist(StorageLevel.MEMORY_AND_DISK)

        sales_metrics, dq_metrics, model_features = build_metrics(enriched_df, invoice_features)

        sales_metrics.repartition(32, "region").write.mode("overwrite").parquet(f"{args.gold_path}/sales_metrics")
        dq_metrics.coalesce(1).write.mode("overwrite").parquet(f"{args.gold_path}/dq_metrics")
        model_features.repartition(64, "region").write.mode("overwrite").parquet(f"{args.gold_path}/model_features")

        sales_metrics.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/tableau/sales_metrics_csv")
        dq_metrics.coalesce(1).write.mode("overwrite").option("header", True).csv("outputs/tableau/dq_metrics_csv")

        if args.save_plan:
            os.makedirs(os.path.dirname(args.plan_output), exist_ok=True)
            with open(args.plan_output, "w", encoding="utf-8") as f:
                f.write(f"join_mode={join_mode}\n")
                f.write("==== ENRICHED_DF PHYSICAL PLAN ====\n")
                f.write(enriched_df._jdf.queryExecution().executedPlan().toString())
                f.write("\n==== SALES_METRICS PHYSICAL PLAN ====\n")
                f.write(sales_metrics._jdf.queryExecution().executedPlan().toString())
                f.write("\n")

        curated_rows = enriched_df.count()
        runtime_sec = time.perf_counter() - run_start

        log_lineage(
            pipeline_name="02_pipeline",
            input_path=args.bronze_path,
            output_path=args.gold_path,
            row_count=curated_rows,
            extra={
                "dataset": "UCI Online Retail",
                "join_mode": join_mode,
                "runtime_sec": f"{runtime_sec:.3f}",
            },
        )

        bronze_df.unpersist()
        enriched_df.unpersist()
        invoice_features.unpersist()

        print(f"Pipeline finished. join_mode={join_mode}, curated_rows={curated_rows}, runtime_sec={runtime_sec:.3f}")

    except Exception as e:
        log_lineage(
            pipeline_name="02_pipeline_failed",
            input_path=args.bronze_path,
            output_path=args.gold_path,
            row_count=0,
            extra={"error": str(e)},
        )
        raise
    finally:
        safe_stop(spark)


if __name__ == "__main__":
    main()
