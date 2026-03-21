import pandas as pd
from lib.config import get_raw_file


def load_datasets():
    """Carica i dataset corretti (pazienti, glicemia, parametri biochimici)."""
    df_patients = pd.read_csv(get_raw_file("Patient_info_corrected.csv"))
    df_glucose = pd.read_csv(get_raw_file("Glucose_measurements_corrected.csv"))
    df_biochem = pd.read_csv(get_raw_file("Biochemical_parameters_corrected.csv"))

    df_glucose["Timestamp"] = pd.to_datetime(df_glucose["Timestamp"])
    df_biochem["Timestamp"] = pd.to_datetime(df_biochem["Timestamp"])

    print(f"Loaded {len(df_patients)} patients, {len(df_glucose)} glucose, {len(df_biochem)} biochem records")

    return df_patients, df_glucose, df_biochem


def merge_patient_data(patient_id, df_glucose, df_biochem):
    """Unisce dati glicemici e biochimici per un paziente tramite backward fill."""
    patient_glucose = df_glucose[df_glucose["Patient_ID"] == patient_id]
    patient_biochem = df_biochem[df_biochem["Patient_ID"] == patient_id]

    if patient_biochem.empty:
        return None

    # Associa ogni misurazione glicemica ai dati biochimici più recenti entro 30 giorni
    merged_data = pd.merge_asof(
        patient_glucose,
        patient_biochem,
        on="Timestamp",
        direction="backward",
        tolerance=pd.Timedelta(days=30),
    )

    merged_data = merged_data.dropna()

    return merged_data


def process_all_patients(df_glucose, df_biochem):
    """Unisce dati glicemici e biochimici per tutti i pazienti."""
    print("\nMerging glucose and biochemical data...")

    patients = df_glucose["Patient_ID"].unique()
    results_list = []

    for i, patient_id in enumerate(patients, 1):
        patient_result = merge_patient_data(patient_id, df_glucose, df_biochem)

        if patient_result is not None:
            results_list.append(patient_result)

    combined_data = pd.concat(results_list, ignore_index=True)

    # Rimuovi colonne Patient_ID duplicate dal merge
    combined_data.drop(columns="Patient_ID_y", inplace=True)
    combined_data.rename(columns={"Patient_ID_x": "Patient_ID"}, inplace=True)

    print(f"Merged data for {len(combined_data)} measurements.")

    return combined_data


def add_static_features(glucose_biochem_data, df_patients):
    """Aggiunge le feature statiche del paziente (età, sesso) al dataset."""
    full_dataset = glucose_biochem_data.merge(df_patients, on="Patient_ID", how="left")

    ordered_cols = ["Patient_ID", "Sex", "Age", "Timestamp", "Measurement"]
    other_cols = [col for col in full_dataset.columns if col not in ordered_cols]
    full_dataset = full_dataset[ordered_cols + other_cols]

    print(f"Final dataset shape: {full_dataset.shape}")

    return full_dataset


def save_final_dataset(dataset, output_path):
    """Salva il dataset arricchito finale."""
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("ADDING FEATURES TO GLUCOSE MEASUREMENTS")
    print("=" * 60)

    df_patients, df_glucose, df_biochem = load_datasets()
    glucose_biochem_data = process_all_patients(df_glucose, df_biochem)
    final_dataset = add_static_features(glucose_biochem_data, df_patients)
    save_final_dataset(
        final_dataset,
        get_raw_file("Glucose_measurements_with_static.csv"),
    )

    print("\nFeature addition completed.")
