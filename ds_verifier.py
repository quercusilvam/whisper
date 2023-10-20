#!/usr/bin/env python3

# This script verifies if created datasets are valid

import os
from datasets import Dataset, Audio
from pydub import AudioSegment

INPUT_DIR = 'audio_chunks'
INPUT_CHUNKS_DIR = os.path.join(INPUT_DIR, 'chunks')
INPUT_DS_DIR = os.path.join(INPUT_DIR, 'ds')


def main():
    """Main function of the script."""
    for d in os.listdir(INPUT_DS_DIR):
        print(f'Read dataset {d}')
        ds = read_dataset_from_dir(os.path.join(INPUT_DS_DIR, d))
        chunks = read_audio_chunks_from_dir(os.path.join(INPUT_CHUNKS_DIR, d))
        print(f'Verify dataset {d}')
        compare_dataset_with_audio_chunks(ds, chunks)


def read_dataset_from_dir(ds_dir):
    """Read dataset from directory."""
    dataset = Dataset.load_from_disk(ds_dir)
    return dataset


def read_audio_chunks_from_dir(audio_chunks_dir):
    """Read audio chunks from dir."""
    output_dict = []
    for f in os.listdir(audio_chunks_dir):
        path = os.path.join(audio_chunks_dir, f)
        # print(f'  Found chunk {path}')
        audio_file = AudioSegment.from_mp3(path)
        d = {'file': path, 'len': len(audio_file)}
        output_dict.append(d)
    return output_dict


def compare_dataset_with_audio_chunks(ds, audio_chunks):
    """Compare if filename and length of dataset is the same as read from audio chunks."""
    for i, d in enumerate(ds):
        if d['file'] != audio_chunks[i]['file']:
            print('  Checking ds {}'.format(d))
            print('  With audio_chunk {}'.format(audio_chunks[i]))
            print('[ERROR] - dataset file {0} is not equal audio chunk file {1}'.format(d['file'], audio_chunks[i]['file']))
        if d['len'] != audio_chunks[i]['len']:
            print('  Checking ds {}'.format(d))
            print('  With audio_chunk {}'.format(audio_chunks[i]))
            print('[ERROR] - dataset len {0} is not equal audio chunk len {1}'.format(d['len'], audio_chunks[i]['len']))
    print('[OK]')


main()
