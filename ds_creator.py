#!/usr/bin/env python3

# This script get

import os
from datasets import Dataset, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

MIN_SILENCE_LEN = 500
SILENCE_THRESH = -45
WHISPER_SAMPLING = 16000

files = ['test_1']
FILE_FORMAT = '.mp3'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'
CHUNK_DIR = OUTPUT_DIR + 'chunks/'
DS_DIR = OUTPUT_DIR + 'ds/'

for f in files:
    print(f'Processing file {f}{FILE_FORMAT}')

    song = AudioSegment.from_mp3(INPUT_DIR + f + FILE_FORMAT)
    song = song.set_frame_rate(WHISPER_SAMPLING)

    print(f'  Split on silence')
    chunks = split_on_silence(
        song,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=True,
        seek_step=50
    )

    # now recombine the chunks so that the parts are at max 30 sec long
    print(f'  Create chunks as long as possible')
    max_length = 30 * 1000
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) + len(chunk) < max_length:
            output_chunks[-1] += chunk
        else:
            output_chunks.append(chunk)

    chunk_file_dir = os.path.join(CHUNK_DIR, f)
    ds_file_dir = os.path.join(DS_DIR, f)
    os.makedirs(chunk_file_dir, exist_ok=True)
    os.makedirs(ds_file_dir, exist_ok=True)

    for file in os.listdir(chunk_file_dir):
        os.remove(os.path.join(chunk_file_dir, file))

    for file in os.listdir(ds_file_dir):
        os.remove(os.path.join(ds_file_dir, file))

    # Process each chunk with your parameters
    print(f'  Process chunks')
    output_chunks_fn = []
    output_chunks_len = []
    output_ds = None

    for i, chunk in enumerate(output_chunks):
        output_chunks_fn.append(chunk_file_dir + '/chunk{0}{1}'.format(i, FILE_FORMAT))
        output_chunks_len.append(len(chunk))
        print('  Exporting chunk{0}. Len {1}'.format(output_chunks_fn[-1], output_chunks_len[-1]))
        chunk.export(
            output_chunks_fn[-1],
            format='mp3'
        )

        chunk_dict = {
            'file': [output_chunks_fn[-1]],
            'audio': [output_chunks_fn[-1]],
            'len': [output_chunks_len[-1]]
        }

        if i == 0:
            output_ds = Dataset.from_dict(chunk_dict).cast_column('audio', Audio(sampling_rate=WHISPER_SAMPLING))
        else:
            ds = Dataset.from_dict(chunk_dict).cast_column('audio', Audio(sampling_rate=WHISPER_SAMPLING))
            output_ds = output_ds.add_item(ds[0])

    output_ds.save_to_disk(ds_file_dir)
