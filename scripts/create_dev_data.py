"""
Generate a reduced dataset for fast development iterations.

Samples DEV_PATIENTS patients (seed=42) and filters all raw CSVs and
split parquet files to only include those patients.
"""

import os
import sys
import json
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lib.config import DEV_PATIENTS, _RAW_DATA_DIR, _DEV_RAW_DATA_DIR, _SPLITS_DIR, _DEV_SPLITS_DIR


def get_dev_patients(n=DEV_PATIENTS, seed=42):
    """Sample N patients from Patient_info.csv."""
    df = pd.read_csv(f"{_RAW_DATA_DIR}/Patient_info.csv")
    # Extract numeric patient IDs (same logic as correct_dataset.py)
    patient_ids = df["Patient_ID"].str[-4:].astype(int)
    sampled = patient_ids.sample(n=n, random_state=seed).tolist()
    print(f"Sampled {n} patients: {sampled}")
    return sampled, df


def filter_raw_csv(filename, patient_col, dev_patients):
    """Filter a raw CSV to only include dev patients, save to dev dir."""
    src = f"{_RAW_DATA_DIR}/{filename}"
    dst = f"{_DEV_RAW_DATA_DIR}/{filename}"

    if not os.path.exists(src):
        print(f"  Skipping {filename} (not found)")
        return

    df = pd.read_csv(src)

    # Handle both raw (string) and corrected (int) Patient_ID formats
    if df[patient_col].dtype == object and df[patient_col].str.startswith("LIB").any():
        numeric_ids = df[patient_col].str[-4:].astype(int)
        df_filtered = df[numeric_ids.isin(dev_patients)]
    else:
        df_filtered = df[df[patient_col].isin(dev_patients)]

    df_filtered.to_csv(dst, index=False)
    print(f"  {filename}: {len(df)} -> {len(df_filtered)} rows")


def filter_parquet_splits(dev_patients):
    """Filter split parquet files to only include dev patients."""
    if not os.path.exists(_SPLITS_DIR):
        print(f"  Split directory {_SPLITS_DIR} not found, skipping")
        return

    os.makedirs(_DEV_SPLITS_DIR, exist_ok=True)

    metadata_path = f"{_SPLITS_DIR}/metadata.json"
    if not os.path.exists(metadata_path):
        print("  metadata.json not found, skipping splits")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    new_metadata = {"X_cols": metadata["X_cols"], "y_cols": metadata["y_cols"]}

    for split_name in ["train", "val", "test"]:
        src = f"{_SPLITS_DIR}/{split_name}_set.parquet"
        dst = f"{_DEV_SPLITS_DIR}/{split_name}_set.parquet"

        if not os.path.exists(src):
            print(f"  {split_name}_set.parquet not found, skipping")
            continue

        df = pd.read_parquet(src)
        df_filtered = df[df["Patient_ID"].isin(dev_patients)]
        df_filtered.to_parquet(dst, index=False)
        new_metadata[f"{split_name}_size"] = len(df_filtered)
        print(f"  {split_name}_set.parquet: {len(df)} -> {len(df_filtered)} rows")

    with open(f"{_DEV_SPLITS_DIR}/metadata.json", "w") as f:
        json.dump(new_metadata, f, indent=2)
    print("  metadata.json regenerated")


def main():
    print("=" * 60)
    print("CREATING DEV DATASET")
    print("=" * 60)
    print()

    # Sample patients
    dev_patients, df_patient_info = get_dev_patients()

    # Create dev directories
    os.makedirs(_DEV_RAW_DATA_DIR, exist_ok=True)

    # Filter raw CSVs
    print("\nFiltering raw CSVs...")
    raw_files = [
        ("Patient_info.csv", "Patient_ID"),
        ("Glucose_measurements.csv", "Patient_ID"),
        ("Biochemical_parameters.csv", "Patient_ID"),
    ]
    for filename, col in raw_files:
        filter_raw_csv(filename, col, dev_patients)

    # Also filter corrected files if they exist
    print("\nFiltering corrected CSVs (if present)...")
    corrected_files = [
        ("Glucose_measurements_corrected.csv", "Patient_ID"),
        ("Patient_info_corrected.csv", "Patient_ID"),
        ("Biochemical_parameters_corrected.csv", "Patient_ID"),
        ("Glucose_measurements_with_static.csv", "Patient_ID"),
    ]
    for filename, col in corrected_files:
        filter_raw_csv(filename, col, dev_patients)

    # Filter split parquet files
    print("\nFiltering split parquet files...")
    filter_parquet_splits(dev_patients)

    print(f"\nDev dataset created in data/dev/")
    print(f"Patients: {dev_patients}")


if __name__ == "__main__":
    main()
