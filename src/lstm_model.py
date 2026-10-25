from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_model(input_shape: tuple[int, int], hidden_size: int = 32) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    hidden = tf.keras.layers.LSTM(hidden_size, dropout=0.1, name="market_state")(inputs)
    output = tf.keras.layers.Dense(1, name="next_return")(hidden)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_lstm(sequences: np.ndarray, targets: np.ndarray, train_mask: np.ndarray,
               epochs: int = 20, batch_size: int = 32, seed: int = 42) -> tuple[tf.keras.Model, dict, dict]:
    tf.keras.utils.set_random_seed(seed)
    model = build_model((sequences.shape[1], sequences.shape[2]))
    x_train, y_train = sequences[train_mask], targets[train_mask]
    x_test, y_test = sequences[~train_mask], targets[~train_mask]
    if len(x_test) == 0:
        raise ValueError("The test split contains no sequences.")
    validation_split = 0.15 if len(x_train) >= 20 else 0.0
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss" if validation_split else "loss", patience=4, restore_best_weights=True)]
    history = model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs,
                        batch_size=batch_size, shuffle=False, verbose=0, callbacks=callbacks).history
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    metrics = {"mae": float(mean_absolute_error(y_test, predictions)),
               "rmse": float(np.sqrt(mean_squared_error(y_test, predictions)))}
    results = {"dates": np.array([], dtype="datetime64[ns]"), "actual": y_test, "predicted": predictions}
    return model, {key: [float(value) for value in values] for key, values in history.items()}, {**metrics, "predictions": results}


def extract_states(model: tf.keras.Model, sequences: np.ndarray) -> np.ndarray:
    extractor = tf.keras.Model(model.input, model.get_layer("market_state").output)
    return extractor.predict(sequences, verbose=0).astype(np.float32)


def save_model(model: tf.keras.Model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
