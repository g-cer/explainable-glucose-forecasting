import pandas as pd
import numpy as np
import time
from datetime import datetime


def load_data(file_path):
    """Carica e prepara il dataset delle misurazioni glicemiche."""
    print(f"Loading dataset '{file_path}'...")
    df_glucose = pd.read_csv(file_path)

    df_glucose["Timestamp"] = pd.to_datetime(
        df_glucose["Measurement_date"] + " " + df_glucose["Measurement_time"]
    )

    df_glucose["Patient_ID"] = df_glucose["Patient_ID"].str[-4:]

    print(f"Dataset loaded successfully. Shape: {df_glucose.shape}")
    print(f"Unique patients: {df_glucose['Patient_ID'].nunique()}")
    return df_glucose


def resample_glucose_data(df, freq="15min", tolerance="7min"):
    """Ricampiona le misurazioni glicemiche per paziente con frequenza regolare.

    Args:
        df: DataFrame con le misurazioni glicemiche.
        freq: Frequenza di ricampionamento.
        tolerance: Tolleranza massima per il matching temporale.

    Returns:
        Lista di DataFrame ricampionati, uno per paziente.
    """
    print(f"Starting resampling with frequency {freq} and tolerance {tolerance}...")

    df = df.sort_values(["Patient_ID", "Timestamp"])  # necessario per merge_asof
    resampled_list = []
    total_patients = df["Patient_ID"].nunique()

    for i, (patient_id, group) in enumerate(df.groupby("Patient_ID"), 1):
        if i % 10 == 0 or i == total_patients:
            print(f"Processing patient {patient_id}... ({i}/{total_patients})")

        group = group.set_index("Timestamp").sort_index()

        start = group.index.min()
        end = group.index.max()

        expected = pd.date_range(start=start, end=end, freq=freq)

        expected_df = pd.DataFrame({"Timestamp": expected})
        group_reset = group[["Measurement"]].reset_index()

        resampled = pd.merge_asof(
            expected_df,
            group_reset,
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )

        resampled["Patient_ID"] = patient_id
        resampled_list.append(resampled)

    print("Resampling completed successfully!")
    return resampled_list


def calculate_patient_statistics(resampled_data):
    """Calcola statistiche per paziente dai dati ricampionati."""
    print("Calculating patient statistics...")

    amount_of_data_per_patient = pd.DataFrame(
        [
            {
                "Patient_ID": df["Patient_ID"].iloc[0],
                "Measurement_count": len(df),
                "Measurement_no_missing_values_count": df["Measurement"].count(),
            }
            for df in resampled_data
        ]
    )

    # Conversione misurazioni in giorni
    minutes_in_hour = 60
    hours_in_day = 24
    samples_separation = 15

    amount_of_data_per_patient["Measurement_no_missing_values_count_days"] = (
        amount_of_data_per_patient["Measurement_no_missing_values_count"]
        * samples_separation
        / (minutes_in_hour * hours_in_day)
    )

    return amount_of_data_per_patient


def filter_patients_by_days(df_glucose_resampled, patient_stats, min_days=30):
    """Filtra i pazienti con meno di un numero minimo di giorni di dati."""
    print(f"Filtering patients with less than {min_days} days of data...")

    patients_below_threshold = patient_stats.loc[
        patient_stats["Measurement_no_missing_values_count_days"] < min_days,
        "Patient_ID",
    ].tolist()

    print(f"Patients to be removed: {len(patients_below_threshold)}")
    print(f"Remaining patients: {len(patient_stats) - len(patients_below_threshold)}")

    patients_mask = ~df_glucose_resampled["Patient_ID"].isin(patients_below_threshold)
    filtered_data = df_glucose_resampled[patients_mask]

    return filtered_data


def save_preprocessed_data(data, output_path):
    """Salva i dati preprocessati su file CSV."""
    print(f"Saving preprocessed dataset to '{output_path}'...")
    data.to_csv(output_path, index=False)
    print("Dataset saved successfully!")


def print_final_statistics(data):
    """Stampa le statistiche finali del dataset preprocessato."""
    print("\n" + "=" * 50)
    print("PREPROCESSING RESULTS")
    print("=" * 50)
    print(f"Final dataset shape: {data.shape}")
    print(f"Number of patients: {data['Patient_ID'].nunique()}")
    print(f"Missing values: {data['Measurement'].isnull().sum()}")
    print(
        f"Missing values percentage: {data['Measurement'].isnull().sum() / len(data) * 100:.2f}%"
    )
    print("=" * 50)


def remove_out_of_range_measurements(df, min_val=40, max_val=400):
    """Rimuove le misurazioni fuori dal range consentito."""
    initial_count = len(df)
    filtered_df = df[(df["Measurement"] >= min_val) & (df["Measurement"] <= max_val)]
    removed_count = initial_count - len(filtered_df)
    print(
        f"Valori rimossi perché fuori dal range [{min_val}, {max_val}]: {removed_count}"
    )
    return filtered_df


def main():
    """Pipeline principale di preprocessing."""
    start_time = time.time()
    print(f"Starting preprocessing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        from lib.config import get_raw_file
        df_glucose = load_data(get_raw_file("Glucose_measurements.csv"))
        df_glucose = remove_out_of_range_measurements(df_glucose)
        resampled_data = resample_glucose_data(df_glucose)
        df_glucose_resampled = pd.concat(resampled_data, ignore_index=True)
        df_glucose_resampled = df_glucose_resampled[
            ["Patient_ID", "Timestamp", "Measurement"]
        ]

        patient_stats = calculate_patient_statistics(resampled_data)
        preprocessed_data = filter_patients_by_days(df_glucose_resampled, patient_stats)
        save_preprocessed_data(
            preprocessed_data,
            get_raw_file("Glucose_measurements_corrected.csv"),
        )

        print_final_statistics(preprocessed_data)

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )
        print(
            f"Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    main()
