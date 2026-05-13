import sys
import logging
from statistics import mean

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from sklearn.utils import shuffle

from music_gen.utils import load_file, make_log_dir
from music_gen.pre_processing import make_windows

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
logging.basicConfig(format="%(message)s", level=logging.INFO)

WINDOW_SIZE: int = 20
INPUT_SHAPE: tuple[int, int, int, int] = (WINDOW_SIZE, 4, 5, 1)
LEARNING_RATE: float = 0.002
EPOCHS: int = 20
SKIP_STEPS: int = 8
REGULARIZER_RATE: float = 0.001

BEST_VAL: float = 0.0
BEST_PARAMS: list = []


def set_global_variables(
    window_size: int,
    learning_rate: float,
    optimizer_val: tf.keras.optimizers.Optimizer,
) -> None:
    global WINDOW_SIZE, LEARNING_RATE, OPTIMIZER
    WINDOW_SIZE = window_size
    LEARNING_RATE = learning_rate
    OPTIMIZER = optimizer_val


def run() -> None:
    global BEST_VAL, BEST_PARAMS

    voices = load_file("data/F.txt")
    n_notes = voices.max() + 1

    log_dir = make_log_dir()

    split_idx = int(voices.shape[1] * 0.9)
    v_train, v_test = voices[:, :split_idx], voices[:, split_idx:]

    x_train, y_train = make_windows(v_train, WINDOW_SIZE, voices=voices, training=True)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, y_train = x_train[SKIP_STEPS:], y_train[SKIP_STEPS:]
    x_test, y_test = make_windows(v_test, WINDOW_SIZE, voices=voices)
    x_test, y_test = shuffle(x_test, y_test)
    logging.debug("[*] x_train: %s y_train: %s", x_train.shape, y_train.shape)
    logging.debug("[*] x_test: %s y_test: %s", x_test.shape, y_test.shape)

    regularizer = tf.keras.regularizers.L2(l2=REGULARIZER_RATE)

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
    model.add(layers.Dense(n_notes * len(voices), kernel_regularizer=regularizer))
    model.add(layers.Reshape((len(voices), n_notes)))
    model.add(layers.Softmax(axis=-1))

    model.compile(
        optimizer=OPTIMIZER,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

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

    val_acc = mean(hist.history["val_accuracy"])
    print(val_acc)

    if val_acc > BEST_VAL:
        BEST_VAL = val_acc
        BEST_PARAMS = [WINDOW_SIZE, LEARNING_RATE, REGULARIZER_RATE, OPTIMIZER]


OPTIMIZER: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

if __name__ == "__main__":
    sizes = [20, 30, 40]
    rates = [1e-4, 1e-5, 1e-6]
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE),
        tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE),
    ]

    print("starting grid search")
    for size in sizes:
        for rate in rates:
            for opt in optimizers:
                print(f"WINDOW: {size}, LEARNING RATE: {rate}, OPTIMIZER: {opt}")
                set_global_variables(size, rate, opt)
                run()

    print("Grid search finished")
    print("Best results:")
    print(BEST_VAL)
    print(BEST_PARAMS)
