"""
Preprocessing script for Dine Brands dataset.
- Downloads raw CSV from S3.
- Cleans and validates columns.
- Handles missing values and data types.
- Splits into train, validation, and test sets.
- Saves results to /opt/ml/processing output folders.
"""

import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# Logging Configuration
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# ----------------------------
# Constants
# ----------------------------
BASE_DIR = "/opt/ml/processing"
FEATURE_NAMES = [
    "date", "store_id", "store_name", "city", "state", "store_type",
    "item_id", "item_name", "category", "price", "quantity_sold",
    "revenue", "food_cost", "profit", "day_of_week", "month",
    "quarter", "is_weekend", "is_holiday", "temperature", "is_promotion",
    "stock_out", "prep_time", "calories", "is_vegetarian",
]
TARGET_COL = "customer_churn"  # standardize column name

# ----------------------------
# Helper Functions
# ----------------------------
def download_from_s3(s3_uri: str, local_path: str):
    """Downloads a file from S3 URI to a local path."""
    logger.info(f"Downloading file from {s3_uri} to {local_path}")
    s3 = boto3.client("s3")
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    s3.download_file(bucket, key, local_path)
    logger.info("Download complete.")


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic cleaning, missing value handling, and schema validation."""
    logger.info("Cleaning and validating dataset...")

    # Rename columns if header mismatch occurs
    if len(df.columns) == len(FEATURE_NAMES) + 1:
        df.columns = FEATURE_NAMES + [TARGET_COL]
    elif len(df.columns) < len(FEATURE_NAMES) + 1:
        raise ValueError(f"Expected {len(FEATURE_NAMES) + 1} columns, got {len(df.columns)}")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Replace placeholder missing values
    df.replace(["?", "NA", "N/A", "null", "None"], np.nan, inplace=True)
    
    # Drop rows with critical missing columns
    df.dropna(subset=["price", "quantity_sold", TARGET_COL], inplace=True)

    # Convert numeric columns safely
    numeric_cols = ["price", "quantity_sold", "revenue", "food_cost", "profit", "temperature", "prep_time", "calories"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle categorical conversions
    cat_cols = ["store_type", "city", "state", "category", "item_name"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # Convert date columns if applicable
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Fill remaining NaNs
    df.fillna({
        "temperature": df["temperature"].median(),
        "prep_time": df["prep_time"].median(),
        "calories": df["calories"].median(),
    }, inplace=True)
    df.fillna("unknown", inplace=True)

    logger.info(f"Final dataset shape after cleaning: {df.shape}")
    return df


def save_split(df_train, df_val, df_test):
    """Saves train/validation/test CSVs to output directories."""
    os.makedirs(f"{BASE_DIR}/train", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/validation", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/test", exist_ok=True)

    df_train.to_csv(f"{BASE_DIR}/train/train.csv", index=False)
    df_val.to_csv(f"{BASE_DIR}/validation/validation.csv", index=False)
    df_test.to_csv(f"{BASE_DIR}/test/test.csv", index=False)

    logger.info("Data saved successfully to:")
    logger.info(f"  Train: {BASE_DIR}/train/train.csv")
    logger.info(f"  Validation: {BASE_DIR}/validation/validation.csv")
    logger.info(f"  Test: {BASE_DIR}/test/test.csv")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    logger.info("Starting Dine Brands preprocessing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    raw_path = f"{BASE_DIR}/raw.csv"
    download_from_s3(args.input_data, raw_path)

        # --- Read data ---
    df = pd.read_csv(local_path)
    df.columns = feature_names + [target_col]
    logger.info(f"Loaded dataset with shape {df.shape}")
    
    # --- Fix data types ---
    # Convert only actual datetime column(s)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure ID and categorical columns stay as string
    df['store_id'] = df['store_id'].astype(str)
    df['store_name'] = df['store_name'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['item_name'] = df['item_name'].astype(str)
    
    # --- Split train/val/test ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    logger.info(
        f"Split into Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}"
    )
    
    # --- Save outputs (with headers!) ---
    train_df.to_csv(f"{base_dir}/train/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/validation/validation.csv", index=False)
    test_df.to_csv(f"{base_dir}/test/test.csv", index=False)
    
    logger.info("✅ Preprocessing complete. Files written with headers.")
    
