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
```

Train a linear regression baseline:

```bash
uv run python -m music_gen.lr
```

Generated MIDI files are saved to `midi_data/`. TensorBoard logs are
written to `logs/`.

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
    pre_processing.py    Window creation and pitch encoding
    post_processing.py   Note sampling and MIDI export
    utils.py             Data loading and path utilities
    lr.py                Linear regression baseline
    gridsearch.py        Hyperparameter grid search
    transformer.py       Transformer model (planned)
data/
    F.txt                Bach chorale dataset
```

## License

MIT
