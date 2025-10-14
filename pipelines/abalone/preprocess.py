"""
Preprocessing script for Dine Brands dataset.
Splits raw CSV into train, validation, and test sets.
Fixes 'Unknown string format' errors during date parsing.
"""

import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("🚀 Starting Dine Brands preprocessing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    for folder in ["train", "validation", "test"]:
        pathlib.Path(f"{base_dir}/{folder}").mkdir(parents=True, exist_ok=True)

    # --- Download raw dataset from S3 ---
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    local_path = f"{base_dir}/raw.csv"

    s3 = boto3.client("s3")
    logger.info(f"📥 Downloading from s3://{bucket}/{key} → {local_path}")
    s3.download_file(bucket, key, local_path)

    # --- Read data safely ---
    df = pd.read_csv(local_path)
    logger.info(f"✅ Loaded dataset shape: {df.shape}")

    # --- Define expected columns ---
    feature_names = [
        "date", "store_id", "store_name", "city", "state", "store_type",
        "item_id", "item_name", "category", "price", "quantity_sold",
        "revenue", "food_cost", "profit", "day_of_week", "month",
        "quarter", "is_weekend", "is_holiday", "temperature", "is_promotion",
        "stock_out", "prep_time", "calories", "is_vegetarian"
    ]
    target_col = "customer_churn"

    # Fix column naming if needed
    all_cols = feature_names + [target_col]
    if len(df.columns) != len(all_cols):
        logger.warning("⚠️ Column count mismatch. Using safe rename fallback.")
        df.columns = all_cols[: len(df.columns)]
    else:
        df.columns = all_cols

    logger.info(f"📋 Columns: {df.columns.tolist()}")

    # --- Drop duplicate headers accidentally read as rows ---
    df = df[df["store_id"] != "store_id"]

    # --- Explicit type conversion ---
    # Convert only 'date' to datetime safely
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Convert other columns to string (non-date columns)
    string_cols = ["store_id", "store_name", "city", "state", "store_type", "item_id", "item_name", "category"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Ensure numeric fields are valid
    numeric_cols = ["price", "quantity_sold", "revenue", "food_cost", "profit",
                    "temperature", "prep_time", "calories"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Handle missing values ---
    df = df.fillna(0)

    # --- Split dataset ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    logger.info(
        f"📊 Data split → Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}"
    )

    # --- Save outputs ---
    train_df.to_csv(f"{base_dir}/train/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False)

    logger.info("✅ Preprocessing complete. Output written to:")
    logger.info(f"  • Train → {base_dir}/train/train.csv")
    logger.info(f"  • Validation → {base_dir}/validation/validation.csv")
    logger.info(f"  • Test → {base_dir}/test/test.csv")
