import pandas as pd
from lib.config import get_raw_file


def process_patient_info():
    """Elabora il file delle informazioni sui pazienti."""
    print("Processing patient_info...")

    df_patients = pd.read_csv(get_raw_file("Patient_info.csv"))

    df_patients["Patient_ID"] = df_patients["Patient_ID"].str[-4:].astype(int)

    df_patients["Initial_measurement_date"] = pd.to_datetime(
        df_patients["Initial_measurement_date"]
    )
    df_patients["Age"] = (
        df_patients["Initial_measurement_date"].dt.year - df_patients["Birth_year"]
    )

    sex_mapping = {"F": 0, "M": 1}
    df_patients["Sex"] = df_patients["Sex"].map(sex_mapping)

    df_patients = df_patients[["Patient_ID", "Sex", "Age"]]
    df_patients.to_csv(get_raw_file("Patient_info_corrected.csv"), index=False)
    print("✓ Patient info file processed")


def process_biochemical_parameters():
    """Elabora il file dei parametri biochimici."""
    print("Processing biochemical parameters...")

    df_biochem = pd.read_csv(get_raw_file("Biochemical_parameters.csv"))

    df_biochem["Patient_ID"] = df_biochem["Patient_ID"].str[-4:].astype(int)

    # Trasformazione da formato long a wide
    df_biochem_wide = df_biochem.pivot_table(
        index=["Patient_ID", "Reception_date"],
        columns="Name",
        values="Value",
        aggfunc="first",
    ).reset_index()

    df_biochem_wide.columns.name = None

    df_biochem = df_biochem_wide.rename(
        columns={
            "Reception_date": "Timestamp",
            "Glycated hemoglobin (A1c)": "HbA1c",
            "Thyrotropin (TSH)": "TSH",
            "Creatinine": "Creatinine",
            "HDL cholesterol": "HDL",
            "Triglycerides": "Triglycerides",
        }
    )

    df_biochem = df_biochem[
        [
            "Patient_ID",
            "Timestamp",
            "HbA1c",
            "TSH",
            "Creatinine",
            "HDL",
            "Triglycerides",
        ]
    ]

    df_biochem.to_csv(
        get_raw_file("Biochemical_parameters_corrected.csv"), index=False
    )
    print("✓ Biochemical parameters file processed")


if __name__ == "__main__":
    process_patient_info()
    process_biochemical_parameters()
