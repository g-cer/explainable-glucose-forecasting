"""Modulo per il caricamento, la trasformazione e lo split dei dati."""

import os
import json
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from lib.data import (
    categorize_glucose,
    scale_data,
    print_dataset_info,
)

HORIZONS = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
LAG_COLS = ["lag105", "lag90", "lag75", "lag60", "lag45", "lag30", "lag15", "lag0"]
TARGET_COL = "lead30"


def stratified_group_train_val_test_split(
    df,
    group_col="Patient_ID",
    y_col="bgClass",
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    random_state=42,
):
    """Split stratificato che mantiene l'integrità dei gruppi (pazienti).

    Args:
        df: DataFrame di input.
        group_col: Colonna di raggruppamento.
        y_col: Colonna per la stratificazione.
        train_size, val_size, test_size: Proporzioni degli split.
        random_state: Seed per riproducibilità.

    Returns:
        Tupla (train_idx, val_idx, test_idx).
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    y = pd.Categorical(df[y_col]).codes.astype(np.int32)
    groups = df[group_col].values
    N = len(df)

    # Split (train+val) vs test
    outer = StratifiedGroupKFold(
        n_splits=round(1 / test_size), shuffle=True, random_state=random_state
    )
    trainval_idx, test_idx = next(outer.split(np.zeros(N), y, groups))

    # Split train vs val
    df_tv = df.iloc[trainval_idx]
    y_tv = y[trainval_idx]
    groups_tv = groups[trainval_idx]

    inner_val_share = val_size / (train_size + val_size)
    inner = StratifiedGroupKFold(
        n_splits=round(1 / inner_val_share), shuffle=True, random_state=random_state
    )
    tr_idx_in_tv, val_idx_in_tv = next(
        inner.split(np.zeros(len(df_tv)), y_tv, groups_tv)
    )

    # Rimappa agli indici originali
    train_idx = df_tv.index[tr_idx_in_tv].to_numpy()
    val_idx = df_tv.index[val_idx_in_tv].to_numpy()
    test_idx = df.index[test_idx].to_numpy()

    validate_group_split(df, train_idx, val_idx, test_idx, group_col)

    return train_idx, val_idx, test_idx


def validate_group_split(df, train_idx, val_idx, test_idx, group_col):
    """Verifica che i gruppi siano correttamente separati tra gli split."""
    groups = [set(df.loc[idx, group_col]) for idx in [train_idx, val_idx, test_idx]]

    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [("train", "validation"), ("train", "test"), ("validation", "test")]

    for (i, j), (name1, name2) in zip(pairs, names):
        assert groups[i].isdisjoint(groups[j]), f"{name1} and {name2} groups overlap"


def report_props(df, idx, y_col="bgClass"):
    """Restituisce le proporzioni delle classi per un sottoinsieme."""
    counts = df.loc[idx, y_col].value_counts(normalize=True).sort_index()
    return counts.to_dict()


def print_split_info(df, train_idx, val_idx, test_idx):
    """Stampa proporzioni, pazienti e distribuzioni delle classi per split."""
    total = len(df)
    proportions = [len(idx) / total for idx in [train_idx, val_idx, test_idx]]
    print(f"Split proportions (rows): {proportions}")

    print("Number of patients by set:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        unique_patients = df.loc[idx, "Patient_ID"].nunique()
        print(f"  {name:<5}: {unique_patients} patients")

    print("Class distributions by set:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        props = report_props(df, idx)
        print(f"  {name:<5}: {props}")


def save_splits(
    train_set, val_set, test_set, X_cols, y_cols, output_dir=None
):
    """Salva i dataset e i metadati su file."""
    if output_dir is None:
        from lib.config import get_splits_dir
        output_dir = get_splits_dir()

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

    print(f"Datasets saved to {output_dir}/")
    for dataset, name in datasets:
        print(f"{name.capitalize()} set: {len(dataset)} samples")


def load_transform(cgm_data, patient_id, horizons, shift_tolerance):
    """Trasforma i dati CGM di un paziente creando le feature lag/lead.

    Args:
        cgm_data: DataFrame Polars con i dati CGM.
        patient_id: Identificativo del paziente.
        horizons: Orizzonti temporali per le feature.
        shift_tolerance: Tolleranza per il matching temporale.

    Returns:
        DataFrame pandas con le feature finestrate.
    """
    cgm = (
        cgm_data.filter(pl.col("Patient_ID") == patient_id)
        .unique(subset="Timestamp")
        .sort("Timestamp")
    )

    if cgm.height == 0:
        return pd.DataFrame()

    cgm_values = cgm.select(["Timestamp", "Measurement"]).rename(
        {"Measurement": "value"}
    )
    result = cgm.clone()

    for h in horizons:
        feature_name = f"lag{h}" if h > 0 else (f"lead{abs(h)}" if h < 0 else "lag0")

        shifted = (
            cgm.with_columns(
                (pl.col("Timestamp") - pl.duration(minutes=h)).alias("TimestampShift")
            )
            .join_asof(
                cgm_values,
                left_on="TimestampShift",
                right_on="Timestamp",
                tolerance=shift_tolerance,
            )
            .select(["Timestamp", "value"])
            .rename({"value": feature_name})
        )

        result = result.join(shifted, on="Timestamp")

    df = result.drop("Measurement").to_pandas()

    if "lead30" in df.columns:
        df["bgClass"] = df["lead30"].apply(categorize_glucose)

    df["Patient_ID"] = patient_id
    return df.dropna().copy()


def prepare_data(
    horizons=None, scale=True, data_path=None, shift_tolerance="1m"
):
    """Pipeline di preparazione dati: caricamento, feature engineering e split.

    Args:
        horizons: Orizzonti temporali per le feature. Default: HORIZONS.
        scale: Se scalare i dati. Default: True.
        data_path: Percorso alla directory dati. Default da config.
        shift_tolerance: Tolleranza per il matching temporale. Default: '1m'.

    Returns:
        Tupla (train_set, val_set, test_set, X_cols, y_cols).
    """
    if data_path is None:
        from lib.config import get_raw_data_dir
        data_path = get_raw_data_dir()

    if horizons is None:
        horizons = HORIZONS

    print("Processing data...")

    cgm_data = pl.read_csv(f"{data_path}/Glucose_measurements_corrected.csv")
    cgm_data = cgm_data.with_columns(pl.col("Timestamp").str.to_datetime())

    all_subjects = (
        cgm_data.select("Patient_ID").unique().to_pandas()["Patient_ID"].tolist()
    )
    print(f"Found {len(all_subjects)} patients in dataset")

    patient_dfs = []
    for i, subject in enumerate(sorted(all_subjects)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing patient {subject} ({i + 1}/{len(all_subjects)})")

        sub_df = load_transform(cgm_data, subject, horizons, shift_tolerance)
        if not sub_df.empty:
            patient_dfs.append(sub_df)

    df = pd.concat(patient_dfs, ignore_index=True)

    if scale:
        df = scale_data(df, LAG_COLS + [TARGET_COL])

    train_idx, val_idx, test_idx = stratified_group_train_val_test_split(df)
    print_split_info(df, train_idx, val_idx, test_idx)

    datasets = [
        df.loc[idx].reset_index(drop=True) for idx in [train_idx, val_idx, test_idx]
    ]
    train_set, val_set, test_set = datasets

    print_dataset_info(train_set, val_set, test_set, LAG_COLS, [TARGET_COL])

    return train_set, val_set, test_set, LAG_COLS, [TARGET_COL]


if __name__ == "__main__":
    train_set, val_set, test_set, X_cols, y_cols = prepare_data(scale=True)

    save_splits(train_set, val_set, test_set, X_cols, y_cols)

    print("Data preparation and saving completed!")
