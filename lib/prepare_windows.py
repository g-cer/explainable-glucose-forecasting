import os
import json
import polars as pl
import pandas as pd
from lib.data import categorize_glucose, scale_data, print_dataset_info
from lib.split_data import (
    stratified_group_train_val_test_split,
    validate_group_split,
    print_split_info,
)

HORIZONS = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
LAG_COLS = ["lag105", "lag90", "lag75", "lag60", "lag45", "lag30", "lag15", "lag0"]
TARGET_COL = "lead30"
SHIFT_TOLERANCE = "1m"


def load_glucose_with_biochemical_features(file_path):
    """Carica le misurazioni glicemiche arricchite con feature biochimiche."""
    df = pl.read_csv(file_path)
    df = df.with_columns(pl.col("Timestamp").str.to_datetime())

    print(f"Dataset loaded. Shape: {df.shape}")

    return df


def create_shifted_feature(cgm_data, horizon, tolerance=SHIFT_TOLERANCE):
    """Crea una singola feature shiftata tramite join_asof.

    Args:
        cgm_data: DataFrame Polars con i dati glicemici.
        horizon: Shift temporale in minuti (positivo=lag, negativo=lead).
        tolerance: Tolleranza per il matching temporale.

    Returns:
        DataFrame con la feature shiftata aggiunta.
    """
    feature_name = (
        f"lag{horizon}"
        if horizon > 0
        else (f"lead{abs(horizon)}" if horizon < 0 else "lag0")
    )

    value_lookup = cgm_data.select(["Timestamp", "Measurement"]).rename(
        {"Measurement": feature_name}
    )

    result = (
        cgm_data.with_columns(
            (pl.col("Timestamp") - pl.duration(minutes=horizon)).alias("TimestampShift")
        )
        .join_asof(
            value_lookup,
            left_on="TimestampShift",
            right_on="Timestamp",
            tolerance=tolerance,
        )
        .drop("TimestampShift")
    )

    return result


def create_windowed_features(cgm_data, patient_id, horizons=HORIZONS):
    """Crea le feature finestrate (lag/lead) per un singolo paziente.

    Args:
        cgm_data: Dataset completo (Polars).
        patient_id: Identificativo del paziente.
        horizons: Orizzonti temporali per le feature.

    Returns:
        DataFrame pandas con le feature finestrate.
    """
    patient_data = (
        cgm_data.filter(pl.col("Patient_ID") == patient_id)
        .unique(subset="Timestamp")
        .sort("Timestamp")
    )

    if patient_data.height == 0:
        return pd.DataFrame()

    result = patient_data.clone()
    glucose_values = patient_data.select(["Timestamp", "Measurement"])

    for horizon in horizons:
        feature_name = (
            f"lag{horizon}"
            if horizon > 0
            else (f"lead{abs(horizon)}" if horizon < 0 else "lag0")
        )

        shifted_values = (
            glucose_values.with_columns(
                (pl.col("Timestamp") - pl.duration(minutes=horizon)).alias(
                    "TimestampShift"
                )
            )
            .join_asof(
                glucose_values.rename({"Measurement": feature_name}),
                left_on="TimestampShift",
                right_on="Timestamp",
                tolerance=SHIFT_TOLERANCE,
            )
            .select(["Timestamp", feature_name])
        )

        result = result.join(shifted_values, on="Timestamp")

    df = result.to_pandas()
    if TARGET_COL in df.columns:
        df["bgClass"] = df[TARGET_COL].apply(categorize_glucose)

    if "Measurement" in df.columns:
        df = df.drop("Measurement", axis=1)

    return df.dropna().copy()


def save_static_splits(
    train_set, val_set, test_set, X_cols, y_cols, output_dir=None
):
    """Salva i dataset e i metadati nella directory degli split statici."""
    if output_dir is None:
        from lib.config import get_static_splits_dir
        output_dir = get_static_splits_dir()

    os.makedirs(output_dir, exist_ok=True)

    datasets = [(train_set, "train"), (val_set, "val"), (test_set, "test")]
    for dataset, name in datasets:
        dataset.to_parquet(f"{output_dir}/{name}_set.parquet", index=False)

    metadata = {
        "X_cols": X_cols,
        "y_cols": y_cols,
        **{f"{name}_size": len(dataset) for dataset, name in datasets},
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDatasets saved to {output_dir}/")
    for dataset, name in datasets:
        print(f"   {name.capitalize()} set: {len(dataset)} samples")


def prepare_static_windowed_data(
    data_path=None,
    scale=True,
    horizons=HORIZONS,
):
    """Pipeline principale per preparare i dati finestrati con feature biochimiche.

    Args:
        data_path: Percorso alle misurazioni arricchite. Default da config.
        scale: Se scalare le feature glicemiche.
        horizons: Orizzonti temporali per le feature lag/lead.

    Returns:
        Tupla (train_set, val_set, test_set, X_cols, y_cols).
    """
    if data_path is None:
        from lib.config import get_raw_file
        data_path = get_raw_file("Glucose_measurements_with_static.csv")

    print("=" * 60)
    print("PREPARING WINDOWED DATA WITH BIOCHEMICAL FEATURES")
    print("=" * 60)

    enriched_data = load_glucose_with_biochemical_features(data_path)

    all_patients = (
        enriched_data.select("Patient_ID").unique().to_pandas()["Patient_ID"].tolist()
    )
    print(f"\nFound {len(all_patients)} patients in dataset")

    print("\nCreating windowed features for all patients...")
    patient_dfs = []

    for i, patient_id in enumerate(sorted(all_patients)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Processing patient {patient_id} ({i + 1}/{len(all_patients)})")

        patient_df = create_windowed_features(enriched_data, patient_id, horizons)
        if not patient_df.empty:
            patient_dfs.append(patient_df)

    df = pd.concat(patient_dfs, ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

    lag_cols = [col for col in df.columns if "lag" in col]
    biochemical_cols = [
        col
        for col in df.columns
        if col in ["Sex", "Age", "HbA1c", "TSH", "Creatinine", "HDL", "Triglycerides"]
    ]
    static_cols = [col for col in df.columns if col in biochemical_cols]
    X_cols = lag_cols + static_cols
    y_cols = [TARGET_COL]

    print(f"\nFeature summary:")
    print(f"   Lag features ({len(lag_cols)}): {lag_cols}")
    print(f"   Static features ({len(static_cols)}): {static_cols}")
    print(f"   Target column: {y_cols}")

    if scale:
        print("\nScaling glucose features...")
        df = scale_data(df, lag_cols + y_cols)

    print("\nPerforming stratified group split...")
    train_idx, val_idx, test_idx = stratified_group_train_val_test_split(df)
    validate_group_split(df, train_idx, val_idx, test_idx, "Patient_ID")
    print("Group split validation passed.")

    print_split_info(df, train_idx, val_idx, test_idx)

    train_set = df.loc[train_idx].reset_index(drop=True)
    val_set = df.loc[val_idx].reset_index(drop=True)
    test_set = df.loc[test_idx].reset_index(drop=True)

    print_dataset_info(train_set, val_set, test_set, X_cols, y_cols)

    return train_set, val_set, test_set, X_cols, y_cols


if __name__ == "__main__":
    train_set, val_set, test_set, X_cols, y_cols = prepare_static_windowed_data(
        scale=True
    )

    save_static_splits(train_set, val_set, test_set, X_cols, y_cols)

    print("\nWindowed data preparation completed.")
