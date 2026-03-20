import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

SUPPORTED_MODELS = ["mlp", "lstm", "gru"]
RNN_MODELS = ["lstm", "gru"]


def set_seeds(seed):
    """Imposta i seed per la riproducibilità in TensorFlow/Keras."""
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def print_model_summary(model):
    """Stampa l'architettura del modello e il numero di parametri."""
    print("\nModel Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")


def create_model(model_type):
    """Crea un modello in base al tipo specificato."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    if model_type == "mlp":
        return create_mlp_model()
    elif model_type == "lstm":
        return create_lstm_model()
    elif model_type == "gru":
        return create_gru_model()


def create_mlp_model():
    """Crea il modello MLP."""
    return keras.Sequential(
        [
            layers.Input(shape=(8,)),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ],
        name="MLP_Model",
    )


def create_lstm_model():
    """Crea il modello LSTM."""
    return keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            layers.LSTM(75, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(75, return_sequences=False),
            layers.Dense(1),
        ],
        name="LSTM_Model",
    )


def create_gru_model():
    """Crea il modello GRU."""
    return keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            layers.GRU(86, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(86, return_sequences=False),
            layers.Dense(1),
        ],
        name="GRU_Model",
    )


def prepare_data(train_set, val_set, test_set, X_cols, y_cols, model_type):
    """Prepara i dati per l'addestramento, con reshape automatico per RNN."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    datasets = [train_set, val_set, test_set]
    X_arrays = [df[X_cols].values.astype(np.float32) for df in datasets]
    y_arrays = [df[y_cols].values.astype(np.float32) for df in datasets]

    y_arrays = [_ensure_1d(y) for y in y_arrays]

    if model_type in RNN_MODELS:
        X_arrays = [_reshape_for_rnn(X) for X in X_arrays]

    shapes = [
        (X_arrays[0], y_arrays[0], "train"),
        (X_arrays[1], y_arrays[1], "val"),
        (X_arrays[2], y_arrays[2], "test"),
    ]
    print("Data shapes:")
    for X, y, name in shapes:
        print(f"X_{name}: {X.shape}, y_{name}: {y.shape}")

    return tuple(X_arrays + y_arrays)


def _ensure_1d(array):
    """Assicura che l'array sia 1D."""
    return array.flatten() if array.ndim > 1 else array


def _reshape_for_rnn(X):
    """Reshape dei dati per modelli RNN."""
    return X.reshape(X.shape[0], X.shape[1], 1)


def create_callbacks(**kwargs):
    """Crea i callback di training con valori di default."""
    defaults = {
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-5,
        "lr_scheduler": True,
        "lr_reduction_factor": 0.1,
        "lr_scheduler_patience": 3,
        "min_learning_rate": 1e-6,
        "monitor": "val_loss",
        "verbose": 1,
    }

    config = {**defaults, **kwargs}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config["monitor"],
            patience=config["early_stopping_patience"],
            min_delta=config["early_stopping_min_delta"],
            verbose=config["verbose"],
            restore_best_weights=True,
        )
    ]

    if config["lr_scheduler"]:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=config["monitor"],
                factor=config["lr_reduction_factor"],
                patience=config["lr_scheduler_patience"],
                min_lr=config["min_learning_rate"],
                verbose=config["verbose"],
            )
        )

    return callbacks


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=4096,
    initial_learning_rate=0.01,
    models_path="models",
    exp_name="model",
):
    """Addestra il modello con la configurazione specificata."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=keras.losses.Huber(),
        metrics=["mae"],
        steps_per_execution=256,
    )

    callbacks = create_callbacks()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # EarlyStopping con restore_best_weights=True già ripristina i migliori pesi
    model_save_path = f"{models_path}/{exp_name}.weights.h5"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_weights(model_save_path)
    print(f"Best model weights saved to: {model_save_path}")

    return model, history


def predict_in_batches(model, data, model_type, batch_size=256):
    """Esegue le predizioni con reshape automatico in base al tipo di modello."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    if model_type in RNN_MODELS:
        data_reshaped = _reshape_for_rnn(data.values)
    else:
        data_reshaped = data.values

    return model.predict(data_reshaped, batch_size=batch_size, verbose=0)
