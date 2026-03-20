"""Utilità per elaborazione dati, scaling e calcolo metriche."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

HYPO = 70.0
HYPER = 180.0
L_BOUND = 40.0
U_BOUND = 400.0


def categorize_glucose(value):
    """Classifica un valore glicemico in Hypo, Normal o Hyper."""
    if pd.isna(value):
        return np.nan
    elif value < HYPO:
        return "Hypo"
    elif value > HYPER:
        return "Hyper"
    else:
        return "Normal"


def scale_data(df, scale_cols):
    """Scala le colonne specificate nell'intervallo [-1, 1].

    Args:
        df: DataFrame da scalare.
        scale_cols: Colonne da scalare.

    Returns:
        DataFrame con colonne scalate.
    """
    df = df.copy()
    for col in scale_cols:
        df[col] = (2 * (df[col] - L_BOUND) / (U_BOUND - L_BOUND)) - 1
    return df


def rescale_data(df, rescale_cols):
    """Riporta le colonne scalate al range originale.

    Args:
        df: DataFrame da riscalare.
        rescale_cols: Colonne da riscalare.

    Returns:
        DataFrame con colonne riscalate.
    """
    df = df.copy()
    for col in rescale_cols:
        df[col] = ((df[col] + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    return df


def load_splits(splits_dir=None):
    """Carica i dataset e i metadati dagli split salvati.

    Args:
        splits_dir: Directory degli split. Default da config.

    Returns:
        Tupla (train_set, val_set, test_set, X_cols, y_cols).
    """
    if splits_dir is None:
        from lib.config import get_splits_dir
        splits_dir = get_splits_dir()

    datasets = []
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(f"{splits_dir}/{name}_set.parquet")
        datasets.append(df)

    with open(f"{splits_dir}/metadata.json", "r") as f:
        metadata = json.load(f)

    return tuple(datasets + [metadata["X_cols"], metadata["y_cols"]])


def calculate_metrics(df):
    """Calcola MAE, MAPE e RMSE per paziente.

    Args:
        df: DataFrame con colonne 'Patient_ID', 'target' e 'y_pred'.

    Returns:
        Tupla (samples, maes, mapes, rmses).
    """
    samples = 0
    maes, mapes, rmses = [], [], []

    for patient_id in df["Patient_ID"].unique():
        patient_data = df[df["Patient_ID"] == patient_id]
        if patient_data.empty:
            continue

        samples += len(patient_data)
        maes.append(mean_absolute_error(patient_data["target"], patient_data["y_pred"]))
        mapes.append(
            mean_absolute_percentage_error(
                patient_data["target"], patient_data["y_pred"]
            )
            * 100
        )
        rmses.append(
            root_mean_squared_error(patient_data["target"], patient_data["y_pred"])
        )

    return samples, maes, mapes, rmses


def print_results(df):
    """Stampa i risultati di valutazione suddivisi per condizione glicemica."""

    def print_metrics(title, samples, maes, mapes, rmses):
        """Stampa le metriche formattate per una condizione."""
        if title != "Cumulative":
            print("~" * 10)
        print(title)
        print(f"Samples: {samples}")
        if maes:  # Only print if we have data
            print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
            print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
            print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")

    samples, maes, mapes, rmses = calculate_metrics(df)
    print_metrics("Cumulative", samples, maes, mapes, rmses)

    for condition in ["Normal", "Hyper", "Hypo"]:
        condition_df = df[df["bgClass"] == condition]
        samples, maes, mapes, rmses = calculate_metrics(condition_df)
        print_metrics(condition, samples, maes, mapes, rmses)


def print_dataset_info(train_set, val_set, test_set, lag_cols, target_cols):
    """Stampa dimensioni dei dataset e colonne utilizzate."""
    sizes = [len(ds) for ds in [train_set, val_set, test_set]]
    names = ["Train", "Validation", "Test"]

    for name, size in zip(names, sizes):
        print(f"{name} size\t{size}")

    print(f"\nFeature columns\t{lag_cols}")
    print(f"Target columns\t{target_cols}\n")
