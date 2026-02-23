from __future__ import annotations

import argparse
from typing import List, Tuple

from pyspark.ml import Transformer
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_utils import build_spark, log_lineage, safe_stop


class BasketSignalTransformer(Transformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        return (
            dataset.withColumn("avg_item_value", F.col("invoice_total") / F.greatest(F.col("total_units"), F.lit(1)))
            .withColumn("items_per_unit_price", F.col("unique_items") / F.greatest(F.col("avg_unit_price"), F.lit(0.01)))
            .withColumn("log_invoice_total", F.log1p(F.abs(F.col("invoice_total"))))
            .fillna(0, subset=["avg_item_value", "items_per_unit_price", "log_invoice_total"])
        )


def build_training_frame(df: DataFrame) -> DataFrame:
    transformed = BasketSignalTransformer().transform(df)
    idx_region = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")
    idx_country = StringIndexer(inputCol="country", outputCol="country_idx", handleInvalid="keep")

    region_model = idx_region.fit(transformed)
    indexed = region_model.transform(transformed)
    country_model = idx_country.fit(indexed)
    indexed = country_model.transform(indexed)

    encoder = OneHotEncoder(
        inputCols=["region_idx", "country_idx"],
        outputCols=["region_vec", "country_vec"],
        handleInvalid="keep",
    )
    encoded = encoder.fit(indexed).transform(indexed)

    assembled = VectorAssembler(
        inputCols=[
            "invoice_total",
            "total_units",
            "unique_items",
            "avg_unit_price",
            "has_return_line",
            "is_weekend",
            "invoice_hour",
            "shipping_risk_score",
            "avg_item_value",
            "items_per_unit_price",
            "log_invoice_total",
            "region_vec",
            "country_vec",
        ],
        outputCol="features",
        handleInvalid="keep",
    ).transform(encoded)

    return assembled.select("invoice_no", "customer_id", "invoice_ts", "target_repeat_30d", "features")


def temporal_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    scored = df.withColumn("event_ts", F.unix_timestamp("invoice_ts"))
    quantiles = scored.approxQuantile("event_ts", [0.7, 0.85], 0.01)

    if len(quantiles) < 2:
        hashed = scored.withColumn("split_id", F.pmod(F.xxhash64("invoice_no", "customer_id"), F.lit(100)))
        return (
            hashed.filter(F.col("split_id") < 70),
            hashed.filter((F.col("split_id") >= 70) & (F.col("split_id") < 85)),
            hashed.filter(F.col("split_id") >= 85),
        )

    q1, q2 = quantiles
    return (
        scored.filter(F.col("event_ts") <= q1),
        scored.filter((F.col("event_ts") > q1) & (F.col("event_ts") <= q2)),
        scored.filter(F.col("event_ts") > q2),
    )


def stratified_sample(df: DataFrame, ratio: float = 0.9) -> DataFrame:
    return df.sampleBy("target_repeat_30d", fractions={0: ratio, 1: ratio}, seed=42)


def bootstrap_ci(pred_df: DataFrame, n_bootstrap: int = 60) -> Tuple[float, float, float]:
    evaluator = BinaryClassificationEvaluator(labelCol="target_repeat_30d", rawPredictionCol="rawPrediction")
    auc_scores: List[float] = []

    for i in range(n_bootstrap):
        sample = pred_df.sample(withReplacement=True, fraction=1.0, seed=42 + i)
        auc_scores.append(float(evaluator.evaluate(sample)))

    scores_df = pred_df.sparkSession.createDataFrame([(x,) for x in auc_scores], ["auc"])
    q = scores_df.approxQuantile("auc", [0.025, 0.5, 0.975], 0.001)
    return float(q[0]), float(q[1]), float(q[2])


def business_value(pred_df: DataFrame) -> float:
    confusion = (
        pred_df.withColumn("label", F.col("target_repeat_30d").cast("int"))
        .withColumn("pred", F.col("prediction").cast("int"))
        .groupBy("label", "pred")
        .count()
    )

    value = 0.0
    for row in confusion.collect():
        l, p, c = int(row["label"]), int(row["pred"]), int(row["count"])
        if l == 1 and p == 1:
            value += 30 * c
        elif l == 0 and p == 1:
            value -= 5 * c
        elif l == 1 and p == 0:
            value -= 20 * c
        else:
            value += 2 * c
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tuned RF model with CI and business metrics")
    parser.add_argument("--features_path", type=str, default="outputs/gold/model_features")
    parser.add_argument("--model_path", type=str, default="outputs/models/mllib_rf_tuned")
    parser.add_argument("--output_path", type=str, default="outputs/tableau/evaluation_csv")
    args = parser.parse_args()

    spark = build_spark("online-retail-evaluation", shuffle_partitions=250)

    try:
        df = build_training_frame(spark.read.parquet(args.features_path))
        train_df, valid_df, test_df = temporal_split(df)

        train_df = stratified_sample(train_df, ratio=0.9)
        valid_df = stratified_sample(valid_df, ratio=0.9)

        model = RandomForestClassificationModel.load(args.model_path)
        pred = model.transform(test_df)

        auc = float(BinaryClassificationEvaluator(labelCol="target_repeat_30d", rawPredictionCol="rawPrediction").evaluate(pred))
        f1 = float(
            MulticlassClassificationEvaluator(labelCol="target_repeat_30d", predictionCol="prediction", metricName="f1").evaluate(
                pred
            )
        )
        ci_low, ci_median, ci_high = bootstrap_ci(pred, n_bootstrap=60)
        expected_profit = business_value(pred)

        rows = [
            ("test_auc", auc),
            ("test_f1", f1),
            ("auc_ci_low_95", ci_low),
            ("auc_ci_median", ci_median),
            ("auc_ci_high_95", ci_high),
            ("expected_profit", expected_profit),
            ("train_count", float(train_df.count())),
            ("valid_count", float(valid_df.count())),
            ("test_count", float(test_df.count())),
        ]

        spark.createDataFrame(rows, ["metric", "value"]).coalesce(1).write.mode("overwrite").option("header", True).csv(
            args.output_path
        )

        log_lineage(
            pipeline_name="05_evaluate",
            input_path=args.features_path,
            output_path=args.output_path,
            row_count=test_df.count(),
            extra={"dataset": "UCI Online Retail", "target": "target_repeat_30d", "bootstrap": "n=60"},
        )

        print("Evaluation complete")

    except Exception as e:
        log_lineage(
            pipeline_name="05_evaluate_failed",
            input_path=args.features_path,
            output_path=args.output_path,
            row_count=0,
            extra={"error": str(e)},
        )
        raise
    finally:
        safe_stop(spark)


if __name__ == "__main__":
    main()
