"""
Preprocessing script for Dine Brands dataset.
Splits raw CSV into train, validation, and test sets.
"""

import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.info("Starting Dine Brands preprocessing...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/train").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/validation").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/test").mkdir(parents=True, exist_ok=True)

    # --- Download raw dataset from S3 ---
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    local_path = f"{base_dir}/raw.csv"

    s3 = boto3.client("s3")
    logger.info(f"Downloading raw data from s3://{bucket}/{key} to {local_path}")
    s3.download_file(bucket, key, local_path)

    # --- Read data ---
    feature_names = [
        "date","store_id","store_name","city","state","store_type",
        "item_id","item_name","category","price","quantity_sold",
        "revenue","food_cost","profit","day_of_week","month",
        "quarter","is_weekend","is_holiday","temperature","is_promotion",
        "stock_out","prep_time","calories","is_vegetarian",
    ]
    target_col = "customer churn"

    df = pd.read_csv(local_path)
    df.columns = feature_names + [target_col]
    logger.info(f"Loaded dataset with shape {df.shape}")

    # Inside PreprocessDineBrandsData or preprocessing.py
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='raise')
        except Exception:
            df[col] = df[col].astype(str)

    # --- Split train/val/test ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    logger.info(
        f"Split into Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}"
    )

    # --- Save outputs ---
    train_path = f"{base_dir}/train/train.csv"
    val_path = f"{base_dir}/validation/validation.csv"
    test_path = f"{base_dir}/test/test.csv"

    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    logger.info("Preprocessing complete. Files written to:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Validation: {val_path}")
    logger.info(f"  Test: {test_path}")
