import math
import numpy as np


def make_windows(
    x: np.ndarray,
    window_size: int,
    voices: np.ndarray | None = None,
    training: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if voices is None:
        voices = x.copy()
    n_voices = len(voices)
    n_windows = x.shape[1] - (window_size + 1)

    if not training:
        n_windows = 1

    x_new = np.zeros((n_windows, window_size, n_voices, 5), dtype=np.float32)
    y_new = x[:, window_size + 1:]

    voices_encoded = encode_inputs(x, voices)

    for window in range(n_windows):
        x_new[window, :] = voices_encoded[window:window + window_size]

    return (x_new, one_hot(y_new, voices)) if training else x_new


def one_hot(y: np.ndarray, voices: np.ndarray | None = None) -> np.ndarray:
    if voices is None:
        voices = y.copy()
    n_notes = voices.max() + 1
    encoded = np.zeros((y.shape[1], y.shape[0], n_notes), dtype=np.int8)
    for v_idx, voice in enumerate(y):
        for t_step, note in enumerate(voice):
            encoded[t_step, v_idx, note] = 1
    return encoded


def encode_inputs(x: np.ndarray, voices: np.ndarray) -> np.ndarray:
    encoded = np.zeros((x.shape[1], x.shape[0], 5), dtype=np.float32)

    voice_offsets = np.zeros(len(voices), dtype=np.float32)
    for v_idx, voice in enumerate(voices):
        voice_offsets[v_idx] = _cal_offset(voice)

    for t_step, beat in enumerate(x.transpose()):
        for v_idx, note in enumerate(beat):
            note = int(note)
            if note == 0:
                encoded[t_step, v_idx] = np.zeros(5)
                continue

            voice_norm = 2 * math.log2(note) + voice_offsets[v_idx]
            x_chroma, y_chroma = _chroma_circle(note)
            x_fifth, y_fifth = _fifth_circle(note)

            encoded[t_step, v_idx] = np.array(
                (voice_norm, x_chroma, y_chroma, x_fifth, y_fifth),
                dtype=np.float32,
            )

    return encoded


def _cal_offset(voice: np.ndarray) -> float:
    voice_sort = np.unique(np.sort(voice))
    voice_max = np.amax(voice)
    voice_min = voice_sort[1]

    voice_offset = (2 * math.log2(voice_max) - 2 * math.log2(voice_min)) / 2
    voice_offset = voice_offset - (2 * math.log2(voice_max))

    return voice_offset


def _chroma_circle(note: int) -> tuple[float, float]:
    theta = (2 * math.pi * (note % 12)) / 12
    x = math.cos(theta)
    y = math.sin(theta)
    return x, y


def _fifth_circle(note: int) -> tuple[float, float]:
    theta = (2 * math.pi * ((7 * note) % 12)) / 12
    x = math.cos(theta)
    y = math.sin(theta)
    return x, y
