#!/usr/bin/env python3

# This script get audio file from INPUT_DIR and split it to chunks. Then save output as Huggingface Dataset

import os
import argparse
from datasets import Dataset, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

MIN_SILENCE_LEN = 500
SILENCE_THRESHOLD = -50
MAX_CHUNK_LENGTH = 30 * 1000
WHISPER_SAMPLING = 16000

INPUT_FILES = ['test_1']
FILE_FORMAT = 'mp3'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'audio_chunks/'
CHUNK_DIR = os.path.join(OUTPUT_DIR, 'chunks/')
DS_DIR = os.path.join(OUTPUT_DIR, 'ds/')


def main():
    """Main function of the script"""
    for f in INPUT_FILES:
        create_and_verify_ds(f)


def create_and_verify_ds(filename):
    """Process audio files, create DS based on them and verify it integrity."""
    process_audio_file(filename)
    verify_ds(filename)


def process_audio_file(audio_file):
    """Convert audio files to chunks and dataset"""
    if create_ds_from_input:
        print(f'Processing file {audio_file}.{FILE_FORMAT}')
        audio = load_audio_file(f'{INPUT_DIR}{audio_file}.{FILE_FORMAT}')

        print(f'  Create chunks as long as possible')
        audio_chunks = generate_audio_chunks(audio)

        print(f'  Init ouputs dir')
        chunk_file_dir = os.path.join(CHUNK_DIR, audio_file)
        ds_file_dir = os.path.join(DS_DIR, audio_file)
        init_output_dirs(chunk_file_dir, ds_file_dir)

        # Process each chunk with your parameters
        print(f'  Save chunks')
        chunks_dict_list = save_audio_chunks(audio_chunks, chunk_file_dir)
        print(f'  Save dataset')
        create_and_save_ds(chunks_dict_list, ds_file_dir)
    else:
        print(f'Processing of file {audio_file} skipped')


def load_audio_file(path, frame_rate=WHISPER_SAMPLING):
    """Load audio file from path and return AudioSegment with set frame_rate"""
    audio_file = AudioSegment.from_mp3(path)
    audio_file = audio_file.set_frame_rate(frame_rate)
    return audio_file


def generate_audio_chunks(audio_file, min_silence_len=MIN_SILENCE_LEN,
                          silence_threshold=SILENCE_THRESHOLD,
                          max_chunk_length=MAX_CHUNK_LENGTH):
    """Generate audio chunks that are split on silence and has max length.

    Find silence spots in audio file (with min_silence_len in ms) and use them as breaking points. After that find
    the largest possible chunks that start/ends with silence spot but not exceed max_chunk_length. This way we get
    chunks as close as possible to chunks using in speach recognition whisper tool, and we are sure audio is not
    split in the middle of spoken word
    """
    silent_spots = split_on_silence(
        audio_file,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        keep_silence=True,
        seek_step=50
    )
    output_chunks = [silent_spots[0]]
    for ss in silent_spots[1:]:
        if len(output_chunks[-1]) + len(ss) < max_chunk_length:
            output_chunks[-1] += ss
        else:
            output_chunks.append(ss)
    return output_chunks


def init_output_dirs(chunk_dir_path, ds_dir_path):
    """Create (if not exists) chunk & ds output dirs. Empty dirs if something exists inside."""
    os.makedirs(chunk_dir_path, exist_ok=True)
    os.makedirs(ds_dir_path, exist_ok=True)

    for file in os.listdir(chunk_dir_path):
        os.remove(os.path.join(chunk_dir_path, file))
    for file in os.listdir(ds_dir_path):
        os.remove(os.path.join(ds_dir_path, file))


def save_audio_chunks(chunks, chunk_dir_path, file_format=FILE_FORMAT):
    """Save audio chunks to given dir path.

    Save audio chunks as audio file in given chunk_dir_path with given file_format.
    Return list of dictionaries with details about saved chunks
    """

    output_dict = []
    for i, chunk in enumerate(chunks):
        output_chunk_fn = chunk_dir_path + '/chunk{:04d}.{}'.format(i, file_format)
        chunk.export(output_chunk_fn, format=file_format)
        # final length for file could be different from chunk itself - so we need to read real file
        output_chunk_len = len(AudioSegment.from_mp3(output_chunk_fn))
        print('    Exported {0}. Len {1}'.format(output_chunk_fn, output_chunk_len))
        d = {'file': [output_chunk_fn], 'audio': [output_chunk_fn], 'len': [output_chunk_len]}
        output_dict.append(d)
    return output_dict


def create_and_save_ds(chunks_dict_list, ds_file_dir):
    """Create dataset from given list of dicts."""
    output_ds = None
    for i, chunk in enumerate(chunks_dict_list):
        file = chunk['file']
        print(f'    Adding {file} to Dataset')
        if i == 0:
            output_ds = Dataset.from_dict(chunk).cast_column('audio', Audio())
        else:
            ds = Dataset.from_dict(chunk).cast_column('audio', Audio())
            output_ds = output_ds.add_item(ds[0])

    output_ds.save_to_disk(ds_file_dir)


def verify_ds(dirname):
    """Verify created dataset"""
    if verify_created_ds:
        print(f'  Read dataset {dirname}')
        ds = read_dataset_from_dir(os.path.join(DS_DIR, dirname))
        chunks = read_audio_chunks_from_dir(os.path.join(CHUNK_DIR, dirname))
        print(f'  Verify dataset {dirname}')
        compare_dataset_with_audio_chunks(ds, chunks)
    else:
        print(f'  Verification of {dirname} skipped')


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
            print('  [ERROR] - dataset file {0} is not equal audio chunk file {1}'.format(d['file'], audio_chunks[i]['file']))
        if d['len'] != audio_chunks[i]['len']:
            print('  Checking ds {}'.format(d))
            print('  With audio_chunk {}'.format(audio_chunks[i]))
            print('  [ERROR] - dataset len {0} is not equal audio chunk len {1}'.format(d['len'], audio_chunks[i]['len']))
    print('[OK]')


parser = argparse.ArgumentParser(description='Create Huggingface DS from audio files that can be used by whisper')
parser.add_argument('-ni', '--no_input', action='store_true',
                    help='If set the script will not create audio_chunks from "input" dir')
parser.add_argument('-t', '--create_test', action='store_true',
                    help='If set the script will create test DS from "test" dir with transcription to test WER')
parser.add_argument('-v', '--verification', action='store_true',
                    help='Verify the created DSes')
args = parser.parse_args()

create_ds_from_input = not args.no_input
create_test_ds = args.create_test
verify_created_ds = args.verification

main()
