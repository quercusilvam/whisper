#!/usr/bin/env python3

# This script verifies if created datasets are valid

import os
from datasets import Dataset, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

INPUT_DIR = '../audio_chunks'
INPUT_CHUNKS_DIR = os.path.join(INPUT_DIR, 'chunks')
INPUT_DS_DIR = os.path.join(INPUT_DIR, 'ds')


def main():
    """Main function of the script"""
    all_ds = []
    all_chunks = []
    for d in os.listdir(INPUT_DS_DIR):
        print(f'Found dataset {d}')
        all_ds.append(read_dataset_from_dir(os.path.join(INPUT_DS_DIR, d)))

    for d in os.listdir(INPUT_CHUNKS_DIR):
        print(f'Found audio chunks {d}')
        all_chunks.append(read_audio_chunks_from_dir(os.path.join(INPUT_CHUNKS_DIR, d)))


def read_dataset_from_dir(ds_dir):
    """Read dataset from directory."""
    dataset = Dataset.load_from_disk(ds_dir)
    return dataset


def read_audio_chunks_from_dir(audio_chunks_dir):
    """Read audio chunks from dir."""
    output_dict = []
    for f in os.listdir(audio_chunks_dir):
        path = os.path.join(audio_chunks_dir, f)
        print(f'  Found chunk {path}')
        audio_file = AudioSegment.from_mp3(path)
        output_dict.append({'file': [path], 'len': [len(audio_file)]})
    return output_dict

main()
