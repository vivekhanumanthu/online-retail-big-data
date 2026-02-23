from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from pyspark.ml import Transformer
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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


def temporal_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    ordered = df.withColumn("event_ts", F.unix_timestamp("invoice_ts"))
    quantiles = ordered.approxQuantile("event_ts", [0.7, 0.85], 0.01)
    if len(quantiles) == 2:
        q1, q2 = quantiles
        return (
            ordered.filter(F.col("event_ts") <= q1),
            ordered.filter((F.col("event_ts") > q1) & (F.col("event_ts") <= q2)),
            ordered.filter(F.col("event_ts") > q2),
        )

    hashed = ordered.withColumn("split_id", F.pmod(F.xxhash64("invoice_no", "customer_id"), F.lit(100)))
    return (
        hashed.filter(F.col("split_id") < 70),
        hashed.filter((F.col("split_id") >= 70) & (F.col("split_id") < 85)),
        hashed.filter(F.col("split_id") >= 85),
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


def evaluate_binary(predictions: DataFrame) -> Dict[str, float]:
    auc_eval = BinaryClassificationEvaluator(labelCol="target_repeat_30d", rawPredictionCol="rawPrediction")
    f1_eval = MulticlassClassificationEvaluator(labelCol="target_repeat_30d", predictionCol="prediction", metricName="f1")
    return {"auc": float(auc_eval.evaluate(predictions)), "f1": float(f1_eval.evaluate(predictions))}


def run_sklearn_baseline(train_df: DataFrame, test_df: DataFrame, output_path: str) -> Dict[str, float]:
    train_sample = train_df.sample(False, 0.2, 42).toPandas()
    test_sample = test_df.sample(False, 0.2, 42).toPandas()

    if train_sample.empty or test_sample.empty:
        return {"sk_lr_auc": 0.0, "sk_lr_f1": 0.0, "sk_rf_auc": 0.0, "sk_rf_f1": 0.0}

    x_train = np.vstack([v.toArray() for v in train_sample["features"]])
    y_train = train_sample["target_repeat_30d"].astype(int)
    x_test = np.vstack([v.toArray() for v in test_sample["features"]])
    y_test = test_sample["target_repeat_30d"].astype(int)

    sk_lr = SkLogisticRegression(max_iter=250)
    sk_lr.fit(x_train, y_train)
    lr_prob = sk_lr.predict_proba(x_test)[:, 1]
    lr_pred = sk_lr.predict(x_test)

    sk_rf = SkRandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    sk_rf.fit(x_train, y_train)
    rf_prob = sk_rf.predict_proba(x_test)[:, 1]
    rf_pred = sk_rf.predict(x_test)

    metrics = {
        "sk_lr_auc": float(roc_auc_score(y_test, lr_prob)),
        "sk_lr_f1": float(f1_score(y_test, lr_pred)),
        "sk_rf_auc": float(roc_auc_score(y_test, rf_prob)),
        "sk_rf_f1": float(f1_score(y_test, rf_pred)),
    }

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/sklearn_models.pkl", "wb") as f:
        pickle.dump({"sk_lr": sk_lr, "sk_rf": sk_rf, "metrics": metrics}, f)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train distributed MLlib models on Online Retail")
    parser.add_argument("--gold_path", type=str, default="outputs/gold")
    parser.add_argument("--model_path", type=str, default="outputs/models")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    args = parser.parse_args()

    spark = build_spark("online-retail-ml", shuffle_partitions=400)
    spark.sparkContext.setCheckpointDir(args.checkpoint_dir)

    try:
        model_df = build_training_frame(spark.read.parquet(f"{args.gold_path}/model_features")).checkpoint(eager=True)
        train_df, valid_df, test_df = temporal_split(model_df)

        lr = LogisticRegression(featuresCol="features", labelCol="target_repeat_30d", maxIter=120, regParam=0.03)
        rf = RandomForestClassifier(featuresCol="features", labelCol="target_repeat_30d", numTrees=160, maxDepth=12)
        gbt = GBTClassifier(featuresCol="features", labelCol="target_repeat_30d", maxIter=70, maxDepth=8)

        lr_model = lr.fit(train_df)
        rf_model = rf.fit(train_df)
        gbt_model = gbt.fit(train_df)

        lr_metrics = evaluate_binary(lr_model.transform(test_df))
        rf_metrics = evaluate_binary(rf_model.transform(test_df))
        gbt_metrics = evaluate_binary(gbt_model.transform(test_df))

        grid = (
            ParamGridBuilder()
            .addGrid(rf.maxDepth, [8, 12, 16])
            .addGrid(rf.numTrees, [80, 140])
            .addGrid(rf.minInstancesPerNode, [1, 4])
            .build()
        )
        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=grid,
            evaluator=BinaryClassificationEvaluator(labelCol="target_repeat_30d", rawPredictionCol="rawPrediction"),
            numFolds=3,
            parallelism=4,
            seed=42,
        )
        cv_model = cv.fit(train_df.unionByName(valid_df))
        tuned_metrics = evaluate_binary(cv_model.bestModel.transform(test_df))

        sklearn_metrics = run_sklearn_baseline(train_df, test_df, args.model_path)

        os.makedirs(args.model_path, exist_ok=True)
        lr_model.write().overwrite().save(f"{args.model_path}/mllib_lr")
        rf_model.write().overwrite().save(f"{args.model_path}/mllib_rf")
        gbt_model.write().overwrite().save(f"{args.model_path}/mllib_gbt")
        cv_model.bestModel.write().overwrite().save(f"{args.model_path}/mllib_rf_tuned")

        metrics_rows: List[Tuple[str, float, float]] = [
            ("mllib_lr", lr_metrics["auc"], lr_metrics["f1"]),
            ("mllib_rf", rf_metrics["auc"], rf_metrics["f1"]),
            ("mllib_gbt", gbt_metrics["auc"], gbt_metrics["f1"]),
            ("mllib_rf_tuned", tuned_metrics["auc"], tuned_metrics["f1"]),
            ("sklearn_lr", sklearn_metrics["sk_lr_auc"], sklearn_metrics["sk_lr_f1"]),
            ("sklearn_rf", sklearn_metrics["sk_rf_auc"], sklearn_metrics["sk_rf_f1"]),
        ]
        spark.createDataFrame(metrics_rows, ["model_name", "auc", "f1"]).coalesce(1).write.mode("overwrite").option(
            "header", True
        ).csv("outputs/tableau/model_metrics_csv")

        log_lineage(
            pipeline_name="03_ml",
            input_path=f"{args.gold_path}/model_features",
            output_path=args.model_path,
            row_count=test_df.count(),
            extra={"dataset": "UCI Online Retail", "target": "target_repeat_30d"},
        )

        print("ML training complete")

    except Exception as e:
        log_lineage(
            pipeline_name="03_ml_failed",
            input_path=f"{args.gold_path}/model_features",
            output_path=args.model_path,
            row_count=0,
            extra={"error": str(e)},
        )
        raise
    finally:
        safe_stop(spark)


if __name__ == "__main__":
    main()
