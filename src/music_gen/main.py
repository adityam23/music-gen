import sys
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from pathlib import Path
from tqdm import tqdm
from sklearn.utils import shuffle

from music_gen.utils import load_file, make_midi_dir, make_log_dir
from music_gen.pre_processing import make_windows
from music_gen.post_processing import NoteSampler, array_to_midi_file

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
logging.basicConfig(format="%(message)s", level=logging.INFO)

WINDOW_SIZE: int = 16
INPUT_SHAPE: tuple[int, int, int, int] = (WINDOW_SIZE, 4, 5, 1)
LEARNING_RATE: float = 0.002
EPOCHS: int = 100
SKIP_STEPS: int = 16 * 16
PREDICT_STEPS: int = 16 * 32


def build_model(n_notes: int, n_voices: int) -> tf.keras.Sequential:
    regularizer = tf.keras.regularizers.L2(l2=0.001)

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=INPUT_SHAPE))
    model.add(
        layers.Conv3D(16, (5, 3, 1), padding="same", kernel_regularizer=regularizer)
    )
    model.add(layers.Reshape((WINDOW_SIZE, -1)))
    model.add(
        layers.LSTM(
            16,
            kernel_regularizer=regularizer,
            dropout=0.5,
            return_sequences=False,
        )
    )
    model.add(layers.Dense(32, kernel_regularizer=regularizer, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, kernel_regularizer=regularizer, activation="relu"))
    model.add(layers.Dense(n_notes * n_voices, kernel_regularizer=regularizer))
    model.add(layers.Reshape((n_voices, n_notes)))
    model.add(layers.Softmax(axis=-1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run() -> None:
    voices = load_file("data/F.txt")
    n_notes = voices.max() + 1

    note_sampler = NoteSampler(voices)

    midi_path = make_midi_dir()
    log_dir = make_log_dir()

    split_idx = int(voices.shape[1] * 0.9)
    v_train, v_test = voices[:, :split_idx], voices[:, split_idx:]

    x_train, y_train = make_windows(v_train, WINDOW_SIZE, voices=voices, training=True)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = x_train[SKIP_STEPS:], y_train[SKIP_STEPS:]
    x_test, y_test = make_windows(v_test, WINDOW_SIZE, voices=voices)
    logging.debug("[*] x_train: %s y_train: %s", x_train.shape, y_train.shape)
    logging.debug("[*] x_test: %s y_test: %s", x_test.shape, y_test.shape)

    model = build_model(n_notes, len(voices))

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=20,
        restore_best_weights=True,
    )
    callbacks = [tensorboard_cb, early_stopping_cb]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    x_pred = np.hstack(
        (voices[:, -WINDOW_SIZE:], np.zeros((len(voices), PREDICT_STEPS)))
    )

    logging.info("[*] predict...")
    for p_step in tqdm(range(PREDICT_STEPS)):
        x_encoded = make_windows(
            x_pred[:, p_step : p_step + WINDOW_SIZE],
            WINDOW_SIZE,
            voices=voices,
            training=False,
        )

        output = model.predict(x_encoded, verbose=0)
        output = output.reshape((output.shape[1], output.shape[2]))

        x_pred_filled = x_pred[:, : p_step + WINDOW_SIZE]
        x_pred_filled = x_pred_filled.transpose()
        next_notes = note_sampler.sample(output, x_pred_filled)

        x_pred[:, WINDOW_SIZE + p_step] = next_notes

    x_pred = x_pred[:, WINDOW_SIZE:]
    x_pred = x_pred.transpose()

    logging.info("[+] predict finished")

    model.save("bach.keras")
    logging.info("[+] saved model")
    np.savetxt("output.txt", x_pred, fmt="%d", delimiter="\t")
    logging.info("[+] saved output")


if __name__ == "__main__":
    run()
