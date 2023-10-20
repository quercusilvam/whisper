#!/usr/bin/env python3

import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio

INPUT_DIR = 'audio_chunks'
INPUT_CHUNKS_DIR = os.path.join(INPUT_DIR, 'chunks')
INPUT_DS_DIR = os.path.join(INPUT_DIR, 'ds')
MODELS = ['openai/whisper-small' ] #"bardsai/whisper-small-pl"


def microseconds_to_audio_timestamp(microseconds):
    """Convert microseconds (int) to string like 00:00:00.0000"""
    hours = microseconds // 3600000
    microseconds -= hours * 3600000
    minutes = microseconds // 60000
    microseconds -= minutes * 60000
    seconds = microseconds // 1000
    microseconds -= seconds * 1000
    return '{:02d}:{:02d}:{:02d}.{:04d}'.format(hours, minutes, seconds, microseconds)


def process_sample(sample, processor, model):
    """Process audio sample, generate and return text."""
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'],
                               return_tensors='pt').input_features
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='pl', task='transcribe')

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return text[0]


for m in MODELS:
    processor = WhisperProcessor.from_pretrained(m)
    model = WhisperForConditionalGeneration.from_pretrained(m)
    model.config.forced_decoder_ids = None

    for dir_name in os.listdir(INPUT_DS_DIR):
        ds = Dataset.load_from_disk(os.path.join(INPUT_DS_DIR, dir_name))

        total_length = 0
        for d in ds:
            l = d['len']
            print(f'{microseconds_to_audio_timestamp(total_length)} -> {microseconds_to_audio_timestamp(total_length + l)}')
            total_length += l
            transcription = process_sample(d['audio'], processor, model)
            print(transcription)
