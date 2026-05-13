"""Generate multi-track MIDI from prediction arrays with proper tempo."""
import numpy as np
from music21 import midi, stream, instrument, tempo, meter, note, duration


def prediction_to_midi(
    data: np.ndarray,
    output_path: str,
    delta_16th: int = 240,
    bpm: int = 100,
) -> None:
    """Convert (4, N) prediction array to a multi-track MIDI file.

    data[0] = soprano, data[1] = alto, data[2] = tenor, data[3] = bass
    """
    instruments = [
        instrument.Flute(),
        instrument.Clarinet(),
        instrument.Violoncello(),
        instrument.Bassoon(),
    ]
    voice_names = ["Soprano", "Alto", "Tenor", "Bass"]

    score = stream.Score()
    score.insert(0, tempo.MetronomeMark(number=bpm))

    for v_idx in range(4):
        part = stream.Part()
        part.insert(0, instruments[v_idx])
        part.partName = voice_names[v_idx]

        voice = data[v_idx]
        splits = np.split(voice, np.where(np.diff(voice))[0] + 1)

        offset = 0.0
        for split in splits:
            note_val = int(split[0])
            note_length = len(split)
            dur = note_length * (60 / bpm) / 4

            if note_val == 0:
                offset += dur
                continue

            midi_pitch = note_val + 7
            n = note.Note(midi_pitch)
            n.duration = duration.Duration(dur)
            n.volume.velocity = 100
            part.insert(offset, n)
            offset += dur

        score.insert(0, part)

    score.write("midi", output_path)
    print(f"  -> {output_path}")


if __name__ == "__main__":
    lstm_data = np.loadtxt("output.txt").T  # (512, 4) -> (4, 512)
    prediction_to_midi(lstm_data, "lstm_combined.mid")

    transformer_data = np.loadtxt("output_transformer.txt").T
    prediction_to_midi(transformer_data, "transformer_combined.mid")
