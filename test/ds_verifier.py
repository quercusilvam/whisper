#!/usr/bin/env python3

# This script verifies if created datasets are valid

import os
from datasets import Dataset, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

INPUT_DIR = '../audio_chunks/ds'


def main():
    """Main function of the script"""
    for d in os.listdir(INPUT_DIR):
        print(f'Found {d}')


main()
