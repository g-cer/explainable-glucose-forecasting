# Explainable Glucose Forecasting

> Explainable Machine Learning for Blood Glucose Prediction in Type 1 Diabetes

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)

## Abstract

Type 1 Diabetes (T1D) requires continuous blood glucose monitoring and timely insulin dosing. Short-term glucose forecasting can help patients anticipate dangerous hypo- and hyperglycemic episodes.

This project implements a reproducible pipeline for **30-minute blood glucose prediction** using the **T1DiabetesGranada** dataset (736 patients, 4 years of CGM measurements). It compares traditional machine learning models (XGBoost, LightGBM, Random Forest) against recurrent neural networks (LSTM, GRU) and a feedforward baseline (MLP). Models are evaluated with standard regression metrics (MAE, RMSE) and the clinically validated **Clarke Error Grid**. Prediction explainability is analyzed through **SHAP** (SHapley Additive exPlanations).

## Research Questions

- **RQ1**: How do neural networks compare to traditional ML for blood glucose prediction?
- **RQ2**: What is the performance gap between a general model and a per-patient model?
- **RQ3**: How explainable are predictions when using XAI techniques (SHAP)?

## Methodology

### Dataset

The [T1DiabetesGranada](https://doi.org/10.1038/s41597-023-02737-4) dataset contains CGM (Continuous Glucose Monitor) readings from **736 patients** collected over **4 years** at the Hospital Universitario San Cecilio de Granada. Readings are resampled to **15-minute intervals** after outlier removal (values outside [40, 400] mg/dL) and patient filtering (minimum 30 days of data).

### Feature Engineering

Each sample is a sliding window of **8 lagged glucose values** at 15-minute intervals (from 105 to 0 minutes before the prediction point). The target is the glucose value **30 minutes ahead**. An optional set of biochemical and demographic features (HbA1c, TSH, Creatinine, HDL, Triglycerides, Sex, Age) is also explored, but shows negligible improvement over glucose-only features (98.5% of feature importance comes from glucose lags).

### Data Splitting

Data is split into **train (70%, 483 patients) / validation (10%, 68 patients) / test (20%, 138 patients)** using a **stratified group split by patient**, ensuring no patient appears in multiple sets. Glucose values are scaled to [-1, 1] using fixed physiological bounds of [40, 400] mg/dL.

### Models

| Type | Models |
|------|--------|
| Traditional ML | Random Forest, XGBoost, LightGBM |
| Neural Networks | MLP (feedforward baseline), LSTM, GRU |

Hyperparameter tuning uses **Optuna** (Bayesian optimization, 40 trials) for gradient boosting models and **grid search** (neuron count, layers, dropout) for RNNs. The best models from each family — **XGBoost** and **GRU** — are selected as representatives for final evaluation.

### Evaluation

- **Regression metrics**: MAE, MAPE, RMSE (computed per-patient, then averaged)
- **Clinical evaluation**: Clarke Error Grid (zones A–E, where A+B = clinically acceptable)
- **Explainability**: SHAP — TreeExplainer for XGBoost, KernelExplainer for GRU

## Results

- **RQ1** — GRU shows a small but consistent advantage over tuned XGBoost (MAE 13.24 vs 13.43 mg/dL, ~1% improvement). Both models achieve **>98% clinically acceptable predictions** (Clarke Error Grid zones A+B). The margin is not sufficient to justify the greater complexity of neural networks in real-world deployment — the choice depends on the trade-off between accuracy and operational simplicity.

- **RQ2** — Personalization is not systematically beneficial: per-patient models improve predictions for patients with regular glycemic profiles, but worsen them for patients with more complex dynamics. Overall, the general model remains preferable for its practicality and robustness across the patient population.

- **RQ3** — SHAP analysis confirms that the most recent glucose readings (lag0, lag15 — the last 30 minutes) dominate predictions for both models, while older lags have near-zero impact. Both XGBoost and GRU learn consistent feature importances. Explanations are physiologically coherent across hypo-, normo-, and hyperglycemia cases, improving model transparency and clinical trust.

## Project Structure

```
lib/            # Reusable Python modules
notebooks/      # 13 sequential pipeline notebooks
scripts/        # Utility scripts
docs/           # Thesis manuscript (thesis.pdf)
data/           # Dataset (not tracked in git)
    T1DiabetesGranada/
    split_sets/
requirements.txt
```

## Pipeline

Notebooks are numbered and should be executed in order.

| #  | Notebook | Description |
|----|----------|-------------|
| 01 | `01_data_exploration.ipynb` | EDA: distribution, ACF/PACF, box plots |
| 02 | `02_preprocessing.ipynb` | Outlier removal, 15-min resampling, patient filtering |
| 03 | `03_split_data.ipynb` | Sliding windows, scaling, stratified split |
| 04 | `04_train_preliminary_ml.ipynb` | Train RF, LightGBM, XGBoost (default params) |
| 05 | `05_train_preliminary_dnn.ipynb` | Train MLP, LSTM, GRU (default params) |
| 06 | `06_preliminary_results.ipynb` | Metrics + Clarke Error Grid on validation set |
| 07 | `07_additional_features.ipynb` | Experiment with biochemical features |
| 08 | `08_tune_ml.ipynb` | Optuna tuning for XGBoost / LightGBM |
| 09 | `09_tune_rnn.ipynb` | Grid search for LSTM / GRU |
| 10 | `10_test_evaluation.ipynb` | Final evaluation: XGBoost + GRU on test set |
| 11 | `11_test_results.ipynb` | Metrics + Clarke Error Grid on test set |
| 12 | `12_gen_vs_pers.ipynb` | General vs per-patient models |
| 13 | `13_shap_explainability.ipynb` | Global and local SHAP analysis |

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | NumPy, Pandas, Polars |
| Deep Learning | TensorFlow / Keras (LSTM, GRU, MLP) |
| Machine Learning | XGBoost, LightGBM, scikit-learn (Random Forest) |
| Explainability (XAI) | SHAP |
| Tuning | Optuna |
| Visualization | Matplotlib, Seaborn |
| Statistics | Statsmodels (ACF/PACF) |

## Reproducibility

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the dataset

The T1DiabetesGranada dataset must be requested through the [official platform](https://doi.org/10.1038/s41597-023-02737-4) and placed in `data/T1DiabetesGranada/`.

### 3. Run the pipeline

Execute notebooks sequentially from `01` to `13`. Each notebook reads its inputs from the outputs of previous steps.

### Google Colab

Notebooks are designed to run on **Google Colab** (GPU recommended for notebooks 05 and 09). Each notebook includes a preamble that mounts Google Drive and configures the project path.

### Dev mode

For quick iteration on a subset of patients, set `DEV_MODE = True` in `lib/config.py`. This uses 20 patients instead of the full 736.

---

## Citation

This project originated as a Bachelor's thesis (*Tesi di Laurea Triennale*) in Computer Science at Università degli Studi di Salerno by Giovanni Cerchia (supervised by Prof. Fabio Palomba, SeSa Lab).

The full manuscript is available at [`docs/thesis.pdf`](docs/thesis.pdf).
