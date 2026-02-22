from __future__ import annotations

import argparse
import os
import ssl
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_utils import build_spark, log_lineage, safe_stop

UCI_ONLINE_RETAIL_ZIP = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

try:
    import certifi
except Exception:
    certifi = None


def download_file(url: str, target_path: str) -> str:
    Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(target_path):
        ssl_context = None
        if certifi is not None:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ssl_context, timeout=120) as resp, open(target_path, "wb") as out:
            out.write(resp.read())
    return target_path


def resolve_source(input_path: str | None, source_url: str, cache_dir: str) -> str:
    if input_path:
        return input_path

    zip_path = os.path.join(cache_dir, "online_retail.zip")
    download_file(source_url, zip_path)
    return zip_path


def extract_xlsx_if_needed(source_path: str, cache_dir: str) -> str:
    lower = source_path.lower()
    if lower.endswith(".xlsx"):
        return source_path

    if lower.endswith(".zip"):
        with zipfile.ZipFile(source_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".xlsx"):
                    out_path = os.path.join(cache_dir, os.path.basename(name))
                    if not os.path.exists(out_path):
                        zf.extract(name, cache_dir)
                        extracted = os.path.join(cache_dir, name)
                        if extracted != out_path:
                            os.replace(extracted, out_path)
                    return out_path

    raise RuntimeError(f"Unsupported input source: {source_path}. Provide a .zip or .xlsx file.")


def normalize_pdf(pdf: pd.DataFrame, ingest_ts: str) -> pd.DataFrame:
    cols = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    }
    pdf = pdf.rename(columns=cols)
    keep_cols = list(cols.values())
    pdf = pdf[keep_cols].copy()

    pdf["invoice_no"] = pdf["invoice_no"].astype(str).str.strip()
    pdf["stock_code"] = pdf["stock_code"].astype(str).str.strip()
    pdf["description"] = pdf["description"].fillna("unknown").astype(str)
    pdf["quantity"] = pd.to_numeric(pdf["quantity"], errors="coerce").fillna(0).astype(int)
    pdf["unit_price"] = pd.to_numeric(pdf["unit_price"], errors="coerce").fillna(0.0)
    pdf["customer_id"] = pdf["customer_id"].fillna(-1).astype("int64").astype(str)
    pdf["country"] = pdf["country"].fillna("unknown").astype(str).str.strip()
    pdf["invoice_date"] = pd.to_datetime(pdf["invoice_date"], errors="coerce")

    pdf = pdf[pdf["invoice_date"].notna()].copy()
    pdf["invoice_date"] = pdf["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pdf["line_total"] = pdf["quantity"] * pdf["unit_price"]
    pdf["is_return"] = (pdf["quantity"] < 0).astype(int)
    pdf["ingested_at_utc"] = ingest_ts
    return pdf


def split_valid_invalid(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    valid_condition = (
        (F.length(F.trim(F.col("invoice_no"))) > 0)
        & (F.length(F.trim(F.col("stock_code"))) > 0)
        & (F.col("quantity") != 0)
        & (F.col("unit_price") > 0)
        & (F.col("customer_id") != "-1")
        & (F.length(F.trim(F.col("country"))) > 0)
    )

    valid_df = (
        df.filter(valid_condition)
        .withColumn("invoice_ts", F.to_timestamp("invoice_date"))
        .withColumn("invoice_year", F.year("invoice_ts"))
        .withColumn("invoice_month", F.month("invoice_ts"))
        .withColumn(
            "transaction_id",
            F.sha2(F.concat_ws("||", "invoice_no", "stock_code", "invoice_date", "customer_id"), 256),
        )
    )

    invalid_df = df.filter(~valid_condition).withColumn(
        "validation_reason",
        F.when(F.col("quantity") == 0, F.lit("quantity_zero"))
        .when(F.col("unit_price") <= 0, F.lit("unit_price_invalid"))
        .when(F.col("customer_id") == "-1", F.lit("missing_customer_id"))
        .otherwise(F.lit("missing_required_field")),
    )

    return valid_df, invalid_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UCI Online Retail data to partitioned Parquet")
    parser.add_argument("--input_path", type=str, default=None, help="Local .zip or .xlsx source")
    parser.add_argument("--source_url", type=str, default=UCI_ONLINE_RETAIL_ZIP)
    parser.add_argument("--cache_dir", type=str, default="data/raw")
    parser.add_argument("--max_records", type=int, default=0, help="0 means all rows")
    parser.add_argument("--bronze_path", type=str, default="outputs/bronze/online_retail")
    parser.add_argument("--quarantine_path", type=str, default="outputs/quarantine/online_retail")
    args = parser.parse_args()

    spark = build_spark("online-retail-ingestion", shuffle_partitions=200)
    os.makedirs(os.path.dirname(args.bronze_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.quarantine_path), exist_ok=True)

    ingest_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        source = resolve_source(args.input_path, args.source_url, args.cache_dir)
        xlsx_path = extract_xlsx_if_needed(source, args.cache_dir)
        pdf = pd.read_excel(xlsx_path, engine="openpyxl")

        if args.max_records > 0:
            pdf = pdf.head(args.max_records)

        pdf = normalize_pdf(pdf, ingest_ts)
        sdf = spark.createDataFrame(pdf)

        valid_df, invalid_df = split_valid_invalid(sdf)
        valid_count = valid_df.count()
        invalid_count = invalid_df.count()

        if valid_count > 0:
            (
                valid_df.write.mode("append")
                .partitionBy("country", "invoice_year", "invoice_month")
                .parquet(args.bronze_path)
            )

        if invalid_count > 0:
            invalid_df.write.mode("append").parquet(args.quarantine_path)

        log_lineage(
            pipeline_name="01_ingest",
            input_path=source,
            output_path=args.bronze_path,
            row_count=valid_count,
            extra={
                "dataset": "UCI Online Retail",
                "raw_rows": str(pdf.shape[0]),
                "invalid_rows": str(invalid_count),
                "storage_format": "parquet_snappy",
            },
        )

        print(
            f"Ingestion finished: raw={pdf.shape[0]}, valid={valid_count}, invalid={invalid_count}, "
            f"bronze_path={args.bronze_path}"
        )

    except Exception as e:
        log_lineage(
            pipeline_name="01_ingest_failed",
            input_path=args.input_path or args.source_url,
            output_path=args.bronze_path,
            row_count=0,
            extra={"error": str(e)},
        )
        raise
    finally:
        safe_stop(spark)


if __name__ == "__main__":
    main()
