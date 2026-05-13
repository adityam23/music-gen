import sys
import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tqdm import tqdm
from sklearn.utils import shuffle

from music_gen.utils import load_file, make_midi_dir
from music_gen.pre_processing import make_windows
from music_gen.post_processing import NoteSampler, array_to_midi_file

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    window_size: int = 16
    conv_filters: int = 16
    conv_kernel: tuple[int, int, int] = (5, 3, 1)
    lstm_units: int = 16
    dense_units: list[int] = field(default_factory=lambda: [32, 128])
    dropout_rate: float = 0.5
    learning_rate: float = 0.002
    l2_reg: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 20
    skip_steps: int = 256
    predict_steps: int = 512
    train_split: float = 0.9
    data_path: str = "data/F.txt"
    output_dir: str = "."
    midi_dir: str = "midi_data"
    log_dir: str = "logs"
    model_path: str = "bach.keras"
    output_file: str = "output.txt"

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (self.window_size, 4, 5, 1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train LSTM model on Bach chorales and generate new music."
    )
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--conv-filters", type=int, default=16)
    parser.add_argument("--lstm-units", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--l2-reg", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-steps", type=int, default=256)
    parser.add_argument("--predict-steps", type=int, default=512)
    parser.add_argument("--data-path", type=str, default="data/F.txt")
    parser.add_argument("--midi-dir", type=str, default="midi_data")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-path", type=str, default="bach.keras")
    parser.add_argument("--output-file", type=str, default="output.txt")

    args = parser.parse_args()
    return Config(
        window_size=args.window_size,
        conv_filters=args.conv_filters,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        skip_steps=args.skip_steps,
        predict_steps=args.predict_steps,
        data_path=args.data_path,
        midi_dir=args.midi_dir,
        log_dir=args.log_dir,
        model_path=args.model_path,
        output_file=args.output_file,
    )


def prepare_data(cfg: Config) -> tuple[np.ndarray, ...]:
    voices = load_file(cfg.data_path)
    n_notes = voices.max() + 1

    split_idx = int(voices.shape[1] * cfg.train_split)
    v_train, v_test = voices[:, :split_idx], voices[:, split_idx:]

    x_train, y_train = make_windows(
        v_train, cfg.window_size, voices=voices, training=True
    )
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = x_train[cfg.skip_steps:], y_train[cfg.skip_steps:]

    x_test, y_test = make_windows(v_test, cfg.window_size, voices=voices)

    logger.info("x_train: %s  y_train: %s", x_train.shape, y_train.shape)
    logger.info("x_test:  %s  y_test:  %s", x_test.shape, y_test.shape)

    return voices, n_notes, x_train, y_train, x_test, y_test


def build_model(n_notes: int, n_voices: int, cfg: Config) -> tf.keras.Sequential:
    n_notes = int(n_notes)
    n_voices = int(n_voices)
    regularizer = tf.keras.regularizers.L2(l2=cfg.l2_reg)

    model = tf.keras.Sequential(name="lstm_bach")
    model.add(layers.InputLayer(input_shape=cfg.input_shape))
    model.add(
        layers.Conv3D(
            cfg.conv_filters,
            cfg.conv_kernel,
            padding="same",
            kernel_regularizer=regularizer,
        )
    )
    model.add(layers.Reshape((cfg.window_size, -1)))
    model.add(
        layers.LSTM(
            cfg.lstm_units,
            kernel_regularizer=regularizer,
            dropout=cfg.dropout_rate,
            return_sequences=False,
        )
    )
    for units in cfg.dense_units:
        model.add(
            layers.Dense(units, kernel_regularizer=regularizer, activation="relu")
        )
        model.add(layers.Dropout(cfg.dropout_rate))
    model.add(layers.Dense(n_notes * n_voices, kernel_regularizer=regularizer))
    model.add(layers.Reshape((n_voices, n_notes)))
    model.add(layers.Softmax(axis=-1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> tf.keras.callbacks.History:
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=cfg.model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=cfg.log_dir,
        histogram_freq=1,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=cfg.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks = [checkpoint_cb, tensorboard_cb, early_stopping_cb, reduce_lr_cb]

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=cfg.epochs,
        callbacks=callbacks,
    )


def generate(
    model: tf.keras.Model,
    voices: np.ndarray,
    note_sampler: NoteSampler,
    cfg: Config,
) -> np.ndarray:
    n_voices = len(voices)
    x_pred = np.hstack(
        (voices[:, -cfg.window_size:], np.zeros((n_voices, cfg.predict_steps)))
    )

    logger.info("[*] generating %d steps...", cfg.predict_steps)
    for p_step in tqdm(range(cfg.predict_steps)):
        x_encoded = make_windows(
            x_pred[:, p_step : p_step + cfg.window_size],
            cfg.window_size,
            voices=voices,
            training=False,
        )

        output = model.predict(x_encoded, verbose=0)
        output = output.reshape((output.shape[1], output.shape[2]))

        x_pred_filled = x_pred[:, : p_step + cfg.window_size]
        x_pred_filled = x_pred_filled.transpose()
        next_notes = note_sampler.sample(output, x_pred_filled)

        x_pred[:, cfg.window_size + p_step] = next_notes

    x_pred = x_pred[:, cfg.window_size:]
    x_pred = x_pred.transpose()

    logger.info("[+] generation finished")
    return x_pred


def save_results(
    prediction: np.ndarray,
    note_sampler: NoteSampler,
    voices: np.ndarray,
    cfg: Config,
) -> None:
    midi_path = make_midi_dir(cfg.midi_dir)
    array_to_midi_file(prediction, midi_path)

    np.savetxt(cfg.output_file, prediction, fmt="%d", delimiter="\t")
    logger.info("[+] saved output to %s", cfg.output_file)
    logger.info("[+] saved midi to %s/", cfg.midi_dir)


def run(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = parse_args()

    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    voices, n_notes, x_train, y_train, x_test, y_test = prepare_data(cfg)
    note_sampler = NoteSampler(voices)

    model = build_model(n_notes, len(voices), cfg)

    history = train_model(model, x_train, y_train, x_test, y_test, cfg)

    best_val_acc = max(history.history["val_accuracy"])
    logger.info("[+] best val accuracy: %.4f", best_val_acc)

    logger.info("[*] loading best model for generation...")
    model = tf.keras.models.load_model(cfg.model_path)

    prediction = generate(model, voices, note_sampler, cfg)
    save_results(prediction, note_sampler, voices, cfg)


if __name__ == "__main__":
    run()
