# music-gen

Bach chorale generation using LSTM and Transformer models trained on
four-voice (SATB) polyphonic music.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:adityam23/music-gen.git
cd music-gen
uv sync
```

## Usage

Train the LSTM model and generate new chorales:

```bash
uv run python -m music_gen.main
uv run python -m music_gen.main --epochs 200 --window-size 32
```

Train the Transformer model:

```bash
uv run python -m music_gen.transformer
uv run python -m music_gen.transformer --epochs 200 --dropout 0.3
```

Train a linear regression baseline:

```bash
uv run python -m music_gen.lr
```

Compare trained models:

```bash
uv run python -m music_gen.evaluate --lstm bach.keras --transformer bach_transformer.keras
```

Grid search hyperparameters:

```bash
uv run python -m music_gen.gridsearch
```

Create combined multi-track MIDI from predictions:

```bash
uv run python make_combined_midi.py
```

Convert combined MIDI to WAV audio (requires timidity++):

```bash
timidity -Ow -o lstm_combined.wav lstm_combined.mid
timidity -Ow -o transformer_combined.wav transformer_combined.mid
```

Generated MIDI files are saved to `midi_data/` and `midi_data_transformer/`.
TensorBoard logs are written to `logs/`.

```bash
uv run tensorboard --logdir logs/
```

## Data

The model trains on `data/F.txt`, a dataset of Bach chorales encoded as
four-part (soprano, alto, tenor, bass) integer pitch sequences.

## Project structure

```
src/
  music_gen/
    main.py              LSTM training and generation pipeline
    transformer.py       Transformer encoder with multi-head attention
    evaluate.py          Model comparison and evaluation harness
    pre_processing.py    Window creation and pitch encoding
    post_processing.py   Note sampling and MIDI export
    utils.py             Data loading and path utilities
    lr.py                Linear regression baseline
    gridsearch.py        Hyperparameter grid search
    make_midi.py         MIDI file export utility
    make_score.py        Score viewer utility
make_combined_midi.py    Multi-track MIDI from prediction arrays
convert_wav.py           Pure Python MIDI-to-WAV synthesis
data/
    F.txt                Bach chorale dataset
tests/
    test_sampler.py      NoteSampler unit test
```

## License

MIT
