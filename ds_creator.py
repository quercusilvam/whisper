#!/usr/bin/env python3

# This script get

import os
from datasets import Dataset, Audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

MIN_SILENCE_LEN = 500
SILENCE_THRESHOLD = -50
MAX_CHUNK_LENGTH = 30 * 1000
WHISPER_SAMPLING = 16000

files = ['test_1']
FILE_FORMAT = '.mp3'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'
CHUNK_DIR = OUTPUT_DIR + 'chunks/'
DS_DIR = OUTPUT_DIR + 'ds/'


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


for f in files:
    print(f'Processing file {f}{FILE_FORMAT}')
    audio = load_audio_file(INPUT_DIR + f + FILE_FORMAT)

    print(f'  Create chunks as long as possible')
    audio_chunks = generate_audio_chunks(audio)

    print(f'  Init ouputs dir')
    chunk_file_dir = os.path.join(CHUNK_DIR, f)
    ds_file_dir = os.path.join(DS_DIR, f)
    init_output_dirs(chunk_file_dir, ds_file_dir)

    # Process each chunk with your parameters
    print(f'  Save chunks')
    output_chunks_fn = []
    output_chunks_len = []
    output_ds = None

    for i, chunk in enumerate(audio_chunks):
        output_chunks_fn.append(chunk_file_dir + '/chunk{0}{1}'.format(i, FILE_FORMAT))
        output_chunks_len.append(len(chunk))
        print('    Exporting {0}. Len {1}'.format(output_chunks_fn[-1], output_chunks_len[-1]))
        chunk.export(
            output_chunks_fn[-1],
            format=FILE_FORMAT
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
