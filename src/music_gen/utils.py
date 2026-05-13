import numpy as np

from pathlib import Path


def load_file(file: str | Path) -> np.ndarray:
    file_path = Path(file)
    voices = np.loadtxt(file_path, dtype=np.int8)
    x_soprano, x_alto, x_tenor, x_bass = [v.flatten() for v in np.hsplit(voices, 4)]
    return np.vstack((x_soprano, x_alto, x_tenor, x_bass))


def make_midi_dir(path: str | Path = "midi_data") -> Path:
    midi_path = Path(path)
    midi_path.mkdir(exist_ok=True)
    return midi_path


def make_log_dir(path: str | Path = "logs") -> Path:
    log_path = Path(path)
    log_path.mkdir(exist_ok=True)
    return log_path
