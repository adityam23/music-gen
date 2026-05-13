import sys
import logging

from music_gen.post_processing import midi_to_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("[!] requires path to midi file as arg")
    else:
        midi_to_score(sys.argv[1])
