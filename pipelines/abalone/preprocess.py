"""
Preprocessing script for Dine Brands dataset.
Splits raw CSV into train, validation, and test sets for SageMaker AutoML.
"""

import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ----------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Dine Brands preprocessing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    input_dir = os.path.join(base_dir, "input")
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Download raw data from S3 if S3 path provided
    # ----------------------------------------------------------------
    input_data = args.input_data
    local_path = os.path.join(base_dir, "raw.csv")

    if input_data.startswith("s3://"):
        bucket = input_data.split("/")[2]
        key = "/".join(input_data.split("/")[3:])
        logger.info(f"📥 Downloading data from s3://{bucket}/{key}")
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_path)
    else:
        logger.info("📄 Using local input file path")
        local_path = input_data

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    df = pd.read_csv(local_path)
    logger.info(f"✅ Loaded dataset with shape {df.shape}")

    # If column names are not present, define them
    expected_features = [
        "date", "store_id", "store_name", "city", "state", "store_type",
        "item_id", "item_name", "category", "price", "quantity_sold",
        "revenue", "food_cost", "profit", "day_of_week", "month",
        "quarter", "is_weekend", "is_holiday", "temperature", "is_promotion",
        "stock_out", "prep_time", "calories", "is_vegetarian", "customer churn"
    ]

    if len(df.columns) == len(expected_features):
        df.columns = expected_features
    else:
        logger.warning("⚠️ Column mismatch — keeping original headers.")

    # ----------------------------------------------------------------
    # Clean up and type conversions
    # ----------------------------------------------------------------
    # Convert date column safely
    # ----------------------------------------------------------------
    # Safe date parsing — prevents "Unknown string format: store_id"
    # ----------------------------------------------------------------
    if "date" in df.columns:
        # Only convert if column has valid date-like values
        def safe_to_datetime(x):
            try:
                return pd.to_datetime(x, errors="raise")
            except Exception:
                return pd.NaT
    
        df["date"] = df["date"].apply(lambda x: safe_to_datetime(x))
    else:
        logger.warning("⚠️ No 'date' column found — skipping datetime conversion.")


    # Convert boolean-like columns
    bool_cols = ["is_weekend", "is_holiday", "is_promotion", "stock_out", "is_vegetarian"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().replace(
                {"true": 1, "false": 0, "yes": 1, "no": 0}
            ).astype(float)

    # Handle numeric columns
    numeric_cols = ["price", "quantity_sold", "revenue", "food_cost", "profit",
                    "temperature", "prep_time", "calories"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no target
    target_col = "customer churn"
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    else:
        raise ValueError("❌ 'customer churn' column missing from dataset.")

    df = df.fillna(0)

    logger.info(f"🧹 Cleaned dataset shape: {df.shape}")

    # ----------------------------------------------------------------
    # Split into train / validation / test
    # ----------------------------------------------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    logger.info(
        f"📊 Split data: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}"
    )

    # ----------------------------------------------------------------
    # Save processed splits
    # ----------------------------------------------------------------
    train_path = f"{base_dir}/train/train.csv"
    val_path = f"{base_dir}/validation/validation.csv"
    test_path = f"{base_dir}/test/test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("✅ Preprocessing complete. Files saved at:")
    logger.info(f"  • Train: {train_path}")
    logger.info(f"  • Validation: {val_path}")
    logger.info(f"  • Test: {test_path}")
