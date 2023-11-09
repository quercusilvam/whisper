#!/usr/bin/env python3

# This script get audio file from INPUT_DIR and split it to chunks. Then save output as Huggingface Dataset

import os
import argparse

from datasets import Dataset, Audio
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from whisperCryptoHelper import WhisperCryptoHelper

MIN_SILENCE_LEN = 500
SILENCE_THRESHOLD = -50
MAX_CHUNK_LENGTH = 30 * 1000
WHISPER_SAMPLING = 16000

AUDIO_FILE_FORMAT = 'mp3'
TRANSCRIPTION_FILE_FORMAT = 'txt'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'audio_chunks/'
CHUNK_DIR = 'chunks/'
DS_DIR = 'ds/'
TEST_DIR = 'test/'


parser = argparse.ArgumentParser(description='Create Huggingface DS from audio files that can be used by whisper')
parser.add_argument('-e', '--encrypt', default=None,
                    help='If set the script will encrypt output datasets using given file as key')
parser.add_argument('-ni', '--no_input', action='store_true',
                    help='If set the script will not create audio_chunks from "input" dir')
parser.add_argument('-t', '--create_test', action='store_true',
                    help='If set the script will create test DS from "test" dir with transcription to test WER')
parser.add_argument('-v', '--verification', action='store_true',
                    help='Verify the created DSes')
args = parser.parse_args()

encrypt_file = args.encrypt
if encrypt_file:
    is_encrypt = True
is_create_ds_from_input = not args.no_input
is_create_test_ds = args.create_test
is_verify_created_ds = args.verification


def main():
    """Main function of the script"""
    create_and_verify_ds()
    create_and_verify_test_ds()


def create_and_verify_ds():
    """Process audio files, create DS based on them and verify it integrity."""
    if is_create_ds_from_input:
        files = Path(INPUT_DIR).glob(f'*.{AUDIO_FILE_FORMAT}')
        for f in list(files):
            print(f'Processing file {f.name}')
            filename = Path(f.name).stem
            process_audio_file(filename)
            verify_ds(os.path.join(OUTPUT_DIR, CHUNK_DIR, filename), os.path.join(OUTPUT_DIR, DS_DIR, filename), filename)
            encrypt_out_ds(os.path.join(OUTPUT_DIR, DS_DIR, filename), os.path.join(OUTPUT_DIR, DS_DIR))
    else:
        print(f'Processing of file {INPUT_DIR} skipped')


def create_and_verify_test_ds():
    """Create test DS based on audio_chunks in 'test' folder and transcription for each chunk."""
    if is_create_test_ds:
        for test_dir in os.scandir(os.path.join(TEST_DIR, CHUNK_DIR)):
            print(f'Create test DS for folder {test_dir.path}')
            dirname = test_dir.name
            process_test_dir(dirname)
            verify_ds(os.path.join(TEST_DIR, CHUNK_DIR, dirname), os.path.join(TEST_DIR, DS_DIR, dirname), dirname)
            encrypt_out_ds(os.path.join(TEST_DIR, DS_DIR, dirname), os.path.join(TEST_DIR, DS_DIR))
    else:
        print(f'Create of test DS for {TEST_DIR} skipped')


def process_audio_file(audio_file):
    """Convert audio files to chunks and dataset"""
    audio = load_audio_file(f'{INPUT_DIR}{audio_file}.{AUDIO_FILE_FORMAT}')

    print(f'  Create chunks as long as possible')
    audio_chunks = generate_audio_chunks(audio)

    print(f'  Init output dir')
    chunk_file_dir = os.path.join(OUTPUT_DIR, CHUNK_DIR, audio_file)
    ds_file_dir = os.path.join(OUTPUT_DIR, DS_DIR, audio_file)
    init_output_dir(chunk_file_dir)
    init_output_dir(ds_file_dir)

    # Process each chunk with your parameters
    print(f'  Save chunks')
    chunks_dict_list = save_audio_chunks(audio_chunks, chunk_file_dir)
    print(f'  Save dataset')
    create_and_save_ds(chunks_dict_list, ds_file_dir)


def load_audio_file(path, frame_rate=WHISPER_SAMPLING, file_format=AUDIO_FILE_FORMAT):
    """Load audio file from path and return AudioSegment with set frame_rate"""
    audio_file = AudioSegment.from_file(path, file_format)
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
    silent_spots = split_to_chunks_shorter_than_max(audio_file, min_silence_len, silence_threshold, max_chunk_length)
    output_chunks = [silent_spots[0]]
    for ss in silent_spots[1:]:
        if len(output_chunks[-1]) + len(ss) < max_chunk_length:
            output_chunks[-1] += ss
        else:
            output_chunks.append(ss)
    return output_chunks


def split_to_chunks_shorter_than_max(audio_file, min_silence_len=MIN_SILENCE_LEN,
                                     silence_threshold=SILENCE_THRESHOLD,
                                     max_chunk_length=MAX_CHUNK_LENGTH):
    """Split audio file on silence to chunks shorter than max_chunk_length.

    If needed change silence_threshold to higher value to split too long chunks
    """
    threshold_step = 5
    silent_spots = split_on_silence(
        audio_file,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        keep_silence=True,
        seek_step=50
    )
    output_chunks = []
    for ss in silent_spots:
        if len(ss) > max_chunk_length and silence_threshold < 0:
            print(f'  [WARNING] Chunk too big. Try splitting with silence_threshold={silence_threshold+threshold_step}')
            for s in split_to_chunks_shorter_than_max(ss,
                                                      min_silence_len,
                                                      silence_threshold+threshold_step,
                                                      max_chunk_length):
                output_chunks.append(s)
        else:
            output_chunks.append(ss)
    return output_chunks


def init_output_dir(dir_path):
    """Create (if not exists) chunk & ds output dirs. Empty dirs if something exists inside."""
    os.makedirs(dir_path, exist_ok=True)

    for file in os.scandir(dir_path):
        if file.is_file():
            os.remove(file.path)


def save_audio_chunks(chunks, chunk_dir_path, file_format=AUDIO_FILE_FORMAT):
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


def verify_ds(chunks_dir, ds_dir, dirname):
    """Verify created dataset"""
    if is_verify_created_ds:
        print(f'  Read dataset {dirname}')
        ds = read_dataset_from_dir(ds_dir)
        chunks = read_audio_chunks_from_dir(chunks_dir)
        print(f'  Verify dataset {dirname}')
        compare_dataset_with_audio_chunks(ds, chunks)
    else:
        print(f'  Verification of {dirname} skipped')


def process_test_dir(dirname):
    """Process test dirname. Create dataset with audio samples and transcription."""
    print(f'  Init output dir')
    ds_file_dir = os.path.join(TEST_DIR, DS_DIR, dirname)
    init_output_dir(ds_file_dir)
    print(f'  Save dataset')
    create_and_save_test_ds(os.path.join(TEST_DIR, CHUNK_DIR, dirname), ds_file_dir)


def create_and_save_test_ds(chunk_dir, ds_file_dir):
    """Create test DS based on audio_chunks with transcription."""
    print(f'  Read chunks from {chunk_dir}')
    chunks = read_audio_chunks_from_dir(chunk_dir)
    print(f'  Read transcription from {chunk_dir}')
    transcriptions = read_transcriptions_from_dir(chunk_dir)
    if len(chunks) == len(transcriptions):
        output_ds = None
        for i, c in enumerate(chunks):
            chunk = {**c, **transcriptions[i]}
            print(chunk)
            if i == 0:
                output_ds = Dataset.from_dict(chunk).cast_column('audio', Audio())
            else:
                ds = Dataset.from_dict(chunk).cast_column('audio', Audio())
                output_ds = output_ds.add_item(ds[0])
        output_ds.save_to_disk(ds_file_dir)
    else:
        print(f'  [ERROR] Length of audio chunks in {chunk_dir} is different than transcriptions number')


def read_dataset_from_dir(ds_dir):
    """Read dataset from directory."""
    dataset = Dataset.load_from_disk(ds_dir)
    return dataset


def read_audio_chunks_from_dir(audio_chunks_dir):
    """Read audio chunks from dir."""
    output_dict = []
    files = Path(audio_chunks_dir).glob(f'*.{AUDIO_FILE_FORMAT}')
    for f in list(files):
        f = f.__str__()
        # print(f'  Found audio chunk {f}')
        audio_file = AudioSegment.from_mp3(f)
        d = {'file': [f], 'audio': [f], 'len': [len(audio_file)]}
        output_dict.append(d)
    return output_dict


def read_transcriptions_from_dir(transcription_chunks_dir):
    """Read transcriptions chunks from dir."""
    output_dict = []
    files = Path(transcription_chunks_dir).glob(f'*.{TRANSCRIPTION_FILE_FORMAT}')
    for f in list(files):
        # print(f'  Found transcription chunk {f}')
        with open(f, 'r') as file:
            transcription = file.readlines()
        d = {'transcription': transcription}
        output_dict.append(d)
    return output_dict


def compare_dataset_with_audio_chunks(ds, audio_chunks):
    """Compare if filename and length of dataset is the same as read from audio chunks."""
    for i, d in enumerate(ds):
        ac_file = audio_chunks[i]['file'][0]
        d_file = d['file']
        if d_file != ac_file:
            print('  Checking ds {}'.format(d))
            print('  With audio_chunk {}'.format(audio_chunks[i]))
            print('  [ERROR] - dataset file {0} is not equal audio chunk file {1}'.format(d_file, ac_file))

        ac_len = audio_chunks[i]['len'][0]
        d_len = d['len']
        if d_len != ac_len:
            print('  Checking ds {}'.format(d))
            print('  With audio_chunk {}'.format(audio_chunks[i]))
            print('  [ERROR] - dataset len {0} is not equal audio chunk len {1}'.format(d_len, ac_len))
    print('[OK]')


def encrypt_out_ds(dataset_path, dest_dir=None):
    """Encrypt created dataset files."""
    if is_encrypt:
        print(f'Encrypt dataset {dataset_path}')
        wch = WhisperCryptoHelper(encrypt_file)
        crypt_file = wch.encrypt_and_zip(dataset_path, dest_dir)
        print(f'  Created encrypted file {crypt_file}')
    else:
        print(f'Skip encryption {dataset_path}')


main()
