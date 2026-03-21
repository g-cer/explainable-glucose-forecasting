# Explainable Glucose Forecasting

> Explainable Machine Learning for Blood Glucose Prediction in Type 1 Diabetes

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)

## Abstract

Type 1 Diabetes (T1D) requires continuous blood glucose monitoring and timely insulin dosing. Short-term glucose forecasting can help patients anticipate dangerous hypo- and hyperglycemic episodes.

This project implements a reproducible pipeline for **30-minute blood glucose prediction** using the **T1DiabetesGranada** dataset (736 patients, 4 years of CGM measurements). It compares traditional machine learning models (XGBoost, LightGBM, Random Forest) against recurrent neural networks (LSTM, GRU) and a feedforward baseline (MLP). Models are evaluated with standard regression metrics (MAE, RMSE) and the clinically validated **Clarke Error Grid**. Prediction explainability is analyzed through **SHAP** (SHapley Additive exPlanations).

## Documents

- [Thesis (PDF)](thesis.pdf) — Full thesis document (in Italian)
- [Defense slides (PDF)](slides.pdf) — Presentation slides (in Italian)

## Research Questions

- **RQ1**: How do neural networks compare to traditional ML for blood glucose prediction?
- **RQ2**: What is the performance gap between a general model and a per-patient model?
- **RQ3**: How explainable are predictions when using XAI techniques (SHAP)?

## Project Structure

The `lib/` directory contains reusable Python modules for preprocessing, model training, and evaluation. The `notebooks/` directory contains 13 numbered notebooks that implement the full pipeline (see below). The dataset in `data/` is not tracked in git.

```
lib/
notebooks/
scripts/
data/
    T1DiabetesGranada/
    split_sets/
requirements.txt
thesis.pdf
slides.pdf
```

## Pipeline

Notebooks are numbered and should be executed in order. Each maps to a section of the thesis.

| #  | Notebook | Description | Thesis §  |
|----|----------|-------------|:---------:|
| 01 | `01_data_exploration.ipynb` | EDA: distribution, ACF/PACF, box plots | 3.2 |
| 02 | `02_preprocessing.ipynb` | Outlier removal, 15-min resampling, patient filtering | 3.3 |
| 03 | `03_split_data.ipynb` | Sliding windows, scaling, stratified split | 3.3 |
| 04 | `04_train_preliminary_ml.ipynb` | Train RF, LightGBM, XGBoost (default params) | 3.4 |
| 05 | `05_train_preliminary_dnn.ipynb` | Train MLP, LSTM, GRU (default params) | 3.4 |
| 06 | `06_preliminary_results.ipynb` | Metrics + Clarke Error Grid on validation set | 3.4 |
| 07 | `07_additional_features.ipynb` | Experiment with biochemical features | 3.5 |
| 08 | `08_tune_ml.ipynb` | Optuna tuning for XGBoost / LightGBM | 3.6 |
| 09 | `09_tune_rnn.ipynb` | Grid search for LSTM / GRU | 3.6 |
| 10 | `10_test_evaluation.ipynb` | Final evaluation: XGBoost + GRU on test set (RQ1) | 4.1 |
| 11 | `11_test_results.ipynb` | Metrics + Clarke Error Grid on test set | 4.1 |
| 12 | `12_gen_vs_pers.ipynb` | General vs per-patient models (RQ2) | 4.2 |
| 13 | `13_shap_explainability.ipynb` | Global and local SHAP analysis (RQ3) | 4.3 |

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

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Google Colab

Notebooks are designed to run on **Google Colab** (GPU recommended). Each notebook includes a preamble that mounts Google Drive and configures the project path.

### Dataset

The T1DiabetesGranada dataset must be requested through the official platform and placed in `data/T1DiabetesGranada/`.

---

## Sintesi

Pipeline riproducibile per la previsione della glicemia a 30 minuti nel Diabete di Tipo 1. Confronta modelli di machine learning tradizionale (XGBoost, LightGBM, Random Forest) con reti neurali ricorrenti (LSTM, GRU) sul dataset T1DiabetesGranada (736 pazienti, 4 anni di misurazioni CGM). Include valutazione clinica tramite Clarke Error Grid e analisi di interpretabilità con SHAP.

Tesi di Laurea Triennale in Informatica — Università degli Studi di Salerno\
Candidato: Giovanni Cerchia
