import sys

from music_gen.utils import load_file, make_midi_dir
from music_gen.post_processing import array_to_midi_file


def run(file_path: str = "data/F.txt") -> None:
    voices = load_file(file_path)
    midi_path = make_midi_dir()
    array_to_midi_file(voices, midi_path)


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/F.txt"
    run(fp)
