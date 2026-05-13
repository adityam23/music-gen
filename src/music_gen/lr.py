import sys
import logging
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from music_gen.utils import load_file, make_midi_dir, make_log_dir
from music_gen.pre_processing import make_windows
from music_gen.post_processing import NoteSampler

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
logging.basicConfig(format="%(message)s", level=logging.INFO)

WINDOW_SIZE: int = 32
SKIP_STEPS: int = 8
PREDICT_STEPS: int = 16 * 32


def run() -> None:
    voices = load_file("data/F.txt")

    note_sampler = NoteSampler(voices)

    midi_path = make_midi_dir()
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

    params = {
        "alpha": [1e-3, 1e-4, 1e-5],
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "lbfg"],
    }

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    x_train_pad = np.array([np.append(item, 1) for item in x_train_flat])

    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)
    x_test_pad = np.array([np.append(item, 1) for item in x_test_flat])

    model = GridSearchCV(
        estimator=Ridge(),
        param_grid=params,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    model.fit(x_train_pad, y_train_flat)

    print(" Results from Grid Search :")
    print("\n The best score across ALL searched params:\n", model.best_score_)
    print(
        "Train Score for Optimized Parameters:      ",
        model.score(x_train_pad, y_train_flat),
    )
    print(
        "Test Score for Optimized Parameters:       ",
        model.score(x_test_pad, y_test_flat),
    )
    print("\n The best estimator across ALL searched params:\n", model.best_estimator_)
    print("\n The best parameters across ALL searched params:\n", model.best_params_)

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
        x_encoded_flat = x_encoded.reshape(x_encoded.shape[0], -1)
        x_encoded_pad = np.array([np.append(item, 1) for item in x_encoded_flat])

        output = model.predict(x_encoded_pad)
        output = np.where(output > 20, 0, output)

        x_pred_filled = x_pred[:, : p_step + WINDOW_SIZE]
        x_pred_filled = x_pred_filled.transpose()
        next_notes = note_sampler.sample(output, x_pred_filled)

        x_pred[:, WINDOW_SIZE + p_step] = next_notes

    x_pred = x_pred[:, WINDOW_SIZE:]
    x_pred = x_pred.transpose()

    logging.info("[+] predict finished")

    np.savetxt("LR_output.txt", x_pred, fmt="%d", delimiter="\t")
    logging.info("[+] saved output")


if __name__ == "__main__":
    run()
