import logging

from tqdm import tqdm

from music_gen.utils import load_file
from music_gen.pre_processing import make_windows
from music_gen.post_processing import NoteSampler, dummy_model_output

logging.basicConfig(format="%(message)s", level=logging.INFO)


def run() -> None:
    voices = load_file("data/F.txt")

    note_sampler = NoteSampler(voices)

    split_idx = int(voices.shape[1] * 0.9)
    _, v_test = voices[:, :split_idx], voices[:, split_idx:]

    window_size = 16
    x_test, y_test = make_windows(v_test, window_size, voices=voices)
    logging.debug("[*] x_test: %s y_test: %s", x_test.shape, y_test.shape)

    correct_pred = 0
    false_pred = 0

    logging.info("[*] test...")
    for p_step in tqdm(range(len(v_test))):
        output = dummy_model_output()

        x_pred_filled = v_test[:, : p_step + window_size]
        x_pred_filled = x_pred_filled.transpose()
        next_notes = note_sampler.sample(output, x_pred_filled)

        if all(v_test[:, p_step + window_size + 1] == next_notes):
            correct_pred += 1
        else:
            false_pred += 1

    logging.info("[+] test finished")

    acc = correct_pred / (correct_pred + false_pred)
    logging.info("[+] sampler accuracy: %.4f", acc)


if __name__ == "__main__":
    run()
