# Previsione interpretabile della glicemia

Pipeline riproducibile per la **previsione della glicemia** in pazienti con diabete di
tipo 1, a partire dai dati di monitoraggio glicemico continuo. Confronta machine learning
tradizionale e reti neurali ricorrenti, valuta i modelli con la Clarke Error Grid oltre che con le
metriche di regressione, e ne analizza le decisioni con SHAP.

> Tesi di Laurea Triennale in Informatica — *Modelli di Machine Learning Interpretabili per la
> Previsione della Glicemia nel Diabete di Tipo 1*, Università degli Studi di Salerno, A.A.
> 2024/2025. Relatore Prof. Fabio Palomba, SeSa Lab.

## Domande di ricerca

- **RQ1** — Come si confrontano le reti neurali con il machine learning tradizionale?
- **RQ2** — Quanto vale la personalizzazione, cioè un modello per paziente rispetto a uno generale?
- **RQ3** — Quanto sono interpretabili le previsioni con tecniche di XAI?

## Approccio

| | |
|---|---|
| **Dataset** | [T1DiabetesGranada](https://doi.org/10.1038/s41597-023-02737-4): 736 pazienti, 4 anni di registrazioni CGM raccolte all'Ospedale Universitario San Cecilio di Granada |
| **Preprocessing** | Rimozione dei valori fuori dall'intervallo fisiologico [40, 400] mg/dL, ricampionamento a 15 minuti, esclusione dei pazienti con meno di 30 giorni di dati |
| **Feature** | Finestra scorrevole di 8 letture glicemiche, da 105 a 0 minuti prima del punto di previsione; target a +30 minuti. Le feature biochimiche e demografiche (HbA1c, TSH, creatinina, HDL, trigliceridi, sesso, età) sono state provate e scartate: il 98,5% dell'importanza resta sui lag glicemici |
| **Split** | 70/10/20 stratificato e raggruppato per paziente, così che nessun paziente compaia in più insiemi. Scaling in [-1, 1] su estremi fisiologici fissi |
| **Modelli** | Random Forest, XGBoost, LightGBM · MLP, LSTM, GRU |
| **Tuning** | Optuna con sampler TPE per i modelli di gradient boosting, grid search a due fasi (neuroni, poi numero di strati, tipo di cella e dropout) per le reti ricorrenti |
| **Valutazione** | MAE, MAPE e RMSE calcolati per paziente e poi mediati; Clarke Error Grid per la validità clinica; SHAP con TreeExplainer per XGBoost e KernelExplainer per GRU |

## Risultati

**RQ1** — La GRU mostra un vantaggio piccolo ma costante sull'XGBoost ottimizzato (MAE 13,24 contro
13,43 mg/dL, circa l'1%). Entrambi superano il 98% di previsioni clinicamente accettabili nelle zone
A+B della Clarke Error Grid. Il margine non basta a giustificare la maggiore complessità di una rete
neurale in produzione: la scelta dipende dal compromesso fra accuratezza e semplicità operativa.

**RQ2** — La personalizzazione non è sistematicamente vantaggiosa. I modelli per paziente migliorano
le previsioni sui profili glicemici regolari e le peggiorano su quelli con dinamiche più complesse.
Nel complesso il modello generale resta preferibile per praticità e robustezza sull'intera
popolazione.

**RQ3** — L'analisi SHAP mostra che le letture più recenti (ultimi 30 minuti) dominano la previsione
in entrambi i modelli, mentre i lag più lontani hanno impatto quasi nullo. XGBoost e GRU apprendono
importanze coerenti fra loro, e le spiegazioni restano fisiologicamente sensate nei casi di ipo-,
normo- e iperglicemia.

I valori riportati provengono dagli esperimenti descritti nella tesi: il repository non versiona
risultati, modelli o grafici. La somma dei pazienti negli split (483 + 68 + 138 = 689) è inferiore ai
736 del dataset perché il filtro sui 30 giorni minimi ne scarta una parte.

## Pipeline

I notebook sono numerati e vanno eseguiti in ordine: ciascuno legge gli output del precedente.

| # | Notebook | Cosa fa |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Analisi esplorativa: distribuzioni, ACF/PACF, boxplot |
| 02 | `02_preprocessing.ipynb` | Rimozione outlier, ricampionamento a 15 minuti, filtro sui pazienti |
| 03 | `03_split_data.ipynb` | Finestre scorrevoli, scaling, split stratificato |
| 04 | `04_train_preliminary_ml.ipynb` | Addestra RF, LightGBM, XGBoost con iperparametri di default |
| 05 | `05_train_preliminary_dnn.ipynb` | Addestra MLP, LSTM, GRU con iperparametri di default |
| 06 | `06_preliminary_results.ipynb` | Metriche e Clarke Error Grid sul validation set |
| 07 | `07_additional_features.ipynb` | Prova con le feature biochimiche e demografiche |
| 08 | `08_tune_ml.ipynb` | Tuning Optuna per XGBoost e LightGBM |
| 09 | `09_tune_rnn.ipynb` | Grid search per LSTM e GRU |
| 10 | `10_test_evaluation.ipynb` | Valutazione finale di XGBoost e GRU sul test set |
| 11 | `11_test_results.ipynb` | Metriche e Clarke Error Grid sul test set |
| 12 | `12_gen_vs_pers.ipynb` | Modello generale contro modelli per paziente |
| 13 | `13_shap_explainability.ipynb` | Analisi SHAP globale e locale |

```
lib/            moduli riutilizzabili (preprocessing, split, finestre, modelli, Clarke Error Grid)
notebooks/      i 13 notebook della pipeline
scripts/        create_dev_data.py — costruisce il sottoinsieme di sviluppo
docs/           thesis.pdf
data/           dataset e split (non versionati)
    T1DiabetesGranada/
    split_sets/
    static_split_sets/
```

## Esecuzione

```bash
pip install -r requirements.txt
```

Il dataset T1DiabetesGranada va richiesto attraverso la
[piattaforma ufficiale](https://doi.org/10.1038/s41597-023-02737-4) e collocato in
`data/T1DiabetesGranada/`. Non è redistribuibile.

I notebook sono pensati per **Google Colab**: ciascuno inizia montando Google Drive e configurando i
percorsi del progetto. Serve una GPU per i notebook 05 e 09, e anche per il **04**, che usa
`cuml.ensemble.RandomForestRegressor` (RAPIDS) e installa RAPIDS in una cella dedicata: scikit-learn
è impiegato solo per le metriche e per lo split stratificato.

Per iterare su un sottoinsieme di 20 pazienti, eseguire prima `python scripts/create_dev_data.py`
per materializzare `data/dev/`, poi impostare `DEV_MODE = True` in `lib/config.py`.

## Documentazione

La tesi completa è in [`docs/thesis.pdf`](docs/thesis.pdf): stato dell'arte, dominio clinico,
progettazione della pipeline, risultati sperimentali per ciascuna domanda di ricerca e limiti del
lavoro.

Un progetto affine sullo stesso dominio, con dataset e metodi diversi:
[diatrend-statistical-analysis](https://github.com/g-cer/diatrend-statistical-analysis).

## Autore

Giovanni Cerchia — Università degli Studi di Salerno.
