"""Convert combined MIDI files to WAV using pure Python synthesis."""
import numpy as np
from scipy.io import wavfile
from music21 import midi, converter
from pathlib import Path

SAMPLE_RATE = 44100
MIDI_A4 = 69
FREQ_A4 = 440.0


def midi_to_freq(note: int) -> float:
    return FREQ_A4 * 2 ** ((note - MIDI_A4) / 12.0)


def synth_note(freq: float, duration: float, velocity: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    n_samples = int(rate * duration)
    if n_samples == 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)

    attack = int(min(rate * 0.01, n_samples))
    decay = int(min(rate * 0.05, n_samples * 0.3))
    release = int(min(rate * 0.05, n_samples * 0.3))
    sustain = n_samples - attack - decay - release

    if sustain < 0:
        attack = min(attack, n_samples // 2)
        sustain = 0
        decay = min(decay, n_samples // 4)
        release = n_samples - attack - decay

    if release < 0:
        release = 0

    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.linspace(1, 0.7, decay),
        np.full(max(sustain, 1), 0.7),
        np.linspace(0.7, 0, max(release, 1)),
    ])[:n_samples]

    return wave * envelope * (velocity / 127.0)


def midi_file_to_wav(midi_path: str, wav_path: str) -> None:
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    tracks_data = []

    for track_idx in range(len(mf.tracks)):
        track = mf.tracks[track_idx]
        events = []
        current_tick = 0
        tempo = 500000  # default 120 BPM
        ticks_per_quarter = mf.ticksPerQuarterNote or 480
        note_ons = {}

        for event in track.events:
            current_tick += event.time
            if event.type == midi.MetaEvents.SET_TEMPO:
                tempo = event.data
            elif event.type == midi.ChannelVoiceMessages.NOTE_ON:
                pitch = event.pitch
                velocity = event.velocity
                if pitch is None:
                    continue
                if velocity is None:
                    velocity = 0
                if velocity > 0:
                    note_ons[pitch] = (current_tick, velocity)
                elif pitch in note_ons:
                    start_tick, vel = note_ons.pop(pitch)
                    note_time_s = (current_tick - start_tick) * tempo / (1_000_000.0 * ticks_per_quarter)
                    freq = midi_to_freq(pitch)
                    events.append((note_time_s, freq, vel))
            elif event.type == midi.ChannelVoiceMessages.NOTE_OFF:
                pitch = event.pitch
                if pitch is None:
                    continue
                if pitch in note_ons:
                    start_tick, vel = note_ons.pop(pitch)
                    note_time_s = (current_tick - start_tick) * tempo / (1_000_000.0 * ticks_per_quarter)
                    freq = midi_to_freq(pitch)
                    events.append((note_time_s, freq, vel))

        if events:
            max_time_s = current_tick * tempo / (1_000_000.0 * ticks_per_quarter)
            track_audio = np.zeros(int(SAMPLE_RATE * max_time_s + 0.5), dtype=np.float32)
            for duration, freq, vel in events:
                note_wave = synth_note(freq, duration, vel)
                offset = 0
                end = min(len(note_wave), len(track_audio))
                track_audio[offset:end] += note_wave[:end]
            tracks_data.append(track_audio)

    if not tracks_data:
        print(f"  skipping {midi_path} (no notes)")
        return

    max_len = max(len(t) for t in tracks_data)
    mixed = np.zeros(max_len, dtype=np.float32)
    for t in tracks_data:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:len(t)] = t
        mixed += padded

    mixed /= max(np.abs(mixed).max(), 1e-8)
    mixed = (mixed * 32767).astype(np.int16)

    wavfile.write(wav_path, SAMPLE_RATE, mixed)


if __name__ == "__main__":
    for name in ["lstm_combined", "transformer_combined"]:
        print(f"Converting {name}...")
        midi_file_to_wav(f"{name}.mid", f"{name}.wav")
        print(f"  -> {name}.wav")
    print("Done.")
