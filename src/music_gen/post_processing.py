import logging
import numpy as np

from pathlib import Path
from music21 import midi, converter


class NoteSampler:
    voices: list[str]
    notes: np.ndarray
    masks: np.ndarray
    freqs: np.ndarray
    starts: np.ndarray
    deltas: np.ndarray
    lengths: np.ndarray
    chords: np.ndarray

    def __init__(self, voices: np.ndarray) -> None:
        x_soprano, x_alto, x_tenor, x_bass = [v.flatten() for v in np.hsplit(voices, 4)]
        indices = np.arange(voices.max() + 1)

        self.voices = ["soprano", "alto", "tenor", "bass"]
        self.notes = indices

        self.masks = np.zeros((len(self.voices), len(self.notes)), dtype=np.int8)
        self.freqs = np.zeros((len(self.voices), len(self.notes)), dtype=np.float32)
        self.starts = np.zeros((len(self.voices), 16), dtype=np.float32)
        self.deltas = np.zeros((len(self.voices), len(self.notes)), dtype=np.float32)

        max_note_length = max(
            len(max(np.split(v, np.where(np.diff(v))[0] + 1), key=len))
            for v in voices  # type: ignore[attr-defined]
        )
        self.lengths = np.zeros((len(self.voices), max_note_length + 1))
        self.chords = np.zeros(
            (len(self.notes), len(self.notes), len(self.notes)), dtype=np.float32
        )

        for v_idx, voice in enumerate([x_soprano, x_alto, x_tenor, x_bass]):
            self.masks[v_idx] = np.isin(self.notes, np.unique(voice))

        bins = np.arange(len(self.notes) + 1)
        for v_idx, voice in enumerate([x_soprano, x_alto, x_tenor, x_bass]):
            self.freqs[v_idx] = np.histogram(voice, bins=bins, density=True)[0]

        for v_idx, voice in enumerate([x_soprano, x_alto, x_tenor, x_bass]):
            splits = np.split(voice, np.where(np.diff(voice))[0] + 1)
            next_pos = 0
            last_note = 0
            for split in splits:
                self.starts[v_idx][next_pos] += 1
                next_pos = (next_pos + len(split)) % 16
                delta: int = 0
                if last_note and split[0]:
                    delta = abs(int(split[0]) - last_note)
                self.deltas[v_idx][delta] += 1
                last_note = int(split[0])
                self.lengths[v_idx][len(split)] += 1
            self.starts[v_idx] /= len(voice) // 16  # type: ignore[operator]
            self.deltas[v_idx] /= np.sum(self.deltas[v_idx])
            self.lengths[v_idx] /= np.sum(self.lengths[v_idx])

        for step in zip(x_soprano, x_alto, x_tenor, x_bass):
            d1, d2, d3 = 0, 0, 0
            if step[0] and step[1]:
                d1 = abs(int(step[0]) - int(step[1]))
            if step[1] and step[2]:
                d2 = abs(int(step[1]) - int(step[2]))
            if step[2] and step[3]:
                d3 = abs(int(step[2]) - int(step[3]))
            self.chords[d1, d2, d3] += 1
        self.chords /= np.sum(self.chords)

    def sample(self, note_probas: np.ndarray, x: np.ndarray) -> np.ndarray:
        time_step = len(x) % 16
        n_iterations = 20
        notes = np.zeros((n_iterations, 4), dtype=np.int8)

        for iteration in range(n_iterations):
            for v_idx, probas in enumerate(note_probas):
                prev_note = x[:, v_idx][-1]

                if self.starts[v_idx][time_step] and np.random.random() <= self.starts[v_idx][time_step]:
                    note = self._sample_voice(probas, voice=v_idx, prior_weight=0.5)
                    delta_note = int(abs(note - prev_note))
                    if note == 0 or prev_note == 0:
                        delta_note = 0

                    while np.random.random() > self.deltas[v_idx][delta_note]:
                        note = self._sample_voice(probas, voice=v_idx, prior_weight=0.5)
                        delta_note = int(abs(note - prev_note))
                        if note == 0 or prev_note == 0:
                            delta_note = 0
                else:
                    note = x[:, v_idx][-1]
                    logging.debug("[*] repeat %s", note)

                notes[iteration, v_idx] = note

        best = 0
        last = 0.0
        max_notes = 0
        for iteration in range(n_iterations):
            scores = np.sum(self.chords[tuple(notes[iteration])])
            if scores > last and len(notes[iteration][notes[iteration] != 0]) >= max_notes:
                last = scores
                best = iteration
                max_notes = len(notes[iteration][notes[iteration] != 0])

        return notes[best]

    def _sample_voice(
        self,
        note_probas: np.ndarray,
        voice: int,
        prior_weight: float,
    ) -> np.int8:
        note_probas = note_probas * self.masks[voice]
        note_probas /= np.sum(note_probas)
        note_probas = note_probas + self.freqs[voice] * prior_weight
        note_probas /= np.sum(note_probas)
        note = np.random.choice(self.notes, p=note_probas)
        logging.debug("[*] Sample note: %s", note)
        return note


def dummy_model_output() -> np.ndarray:
    return np.random.random((4, 77))


def voice_to_midi_track(
    voice: np.ndarray,
    track: int,
    delta_16th: int = 256,
    velocity: int = 100,
) -> midi.MidiTrack:
    mt = midi.MidiTrack(index=track)

    splits = np.split(voice, np.where(np.diff(voice))[0] + 1)

    for split in splits:
        note = int(split[0])
        note += 7 if note else 0
        note_length = len(split)

        note_on = midi.MidiEvent(
            mt, type=midi.ChannelVoiceMessages.NOTE_ON, channel=1
        )
        note_on.pitch = note
        note_on.velocity = velocity if note else 0
        mt.events.append(note_on)

        duration = delta_16th * note_length
        dt = midi.DeltaTime(mt, time=duration)
        mt.events.append(dt)

        note_off = midi.MidiEvent(
            mt, type=midi.ChannelVoiceMessages.NOTE_OFF, channel=1
        )
        note_off.pitch = note
        note_off.velocity = 0
        mt.events.append(note_off)

        dt = midi.DeltaTime(mt, time=0)
        mt.events.append(dt)

    return mt


def array_to_midi_file(
    data: np.ndarray,
    midi_path: str | Path,
) -> None:
    mt_soprano = voice_to_midi_track(data[0], 1)
    mt_alto = voice_to_midi_track(data[1], 2)
    mt_tenor = voice_to_midi_track(data[2], 3)
    mt_bass = voice_to_midi_track(data[3], 4)

    for voice, track in zip(
        ["soprano", "alto", "tenor", "bass"],
        [mt_soprano, mt_alto, mt_tenor, mt_bass],
    ):
        mf = midi.MidiFile()
        mf.tracks.append(track)
        file_name = f"{voice}.mid"
        mf.open(Path(midi_path) / file_name, "wb")
        try:
            mf.write()
        except Exception:
            logging.error("[!] unable to create file %s", file_name)
        finally:
            mf.close()


def midi_to_score(midi_filepath: str | Path) -> None:
    _make_scratch_dir()
    midi_score = converter.parse(midi_filepath, format="midi").chordify()
    midi_score.show()


def midi_to_console(midi_filepath: str | Path) -> None:
    _make_scratch_dir()
    midi_score = converter.parse(midi_filepath, format="midi").chordify()
    midi_score.show("text")


def _make_scratch_dir() -> None:
    scratch_dir = Path("_scratch/_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
