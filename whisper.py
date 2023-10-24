#!/usr/bin/env python3

import os
import evaluate
import gc
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio


INPUT_DIR = 'audio_chunks/'
TEST_DIR = 'test/'
CHUNKS_DIR = 'chunks/'
DS_DIR = 'ds/'
MODELS = ['openai/whisper-small', 'bardsai/whisper-small-pl']


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

    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return text[0]


exit_values=[]
for i, m in enumerate(MODELS):
    print(f'Checking model {m}')

    if i != 0:
        # memory cleanup
        collected = gc.collect()
        print('Garbage collector: collected {:d} objects.'.format(collected))

    processor = WhisperProcessor.from_pretrained(m, language='pl', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(m)

    model.config.forced_decoder_ids = None
    metric = evaluate.load('wer')

    for dir_name in os.scandir(os.path.join(TEST_DIR, DS_DIR)):
        print(f'Read dataset {dir_name.name}')
        ds = Dataset.load_from_disk(dir_name.path)

        total_length = 0
        total_wer = 0
        average_wer = 0
        for i, d in enumerate(ds):
            l = d['len']
            transcription = d['transcription']
            print(f'{microseconds_to_audio_timestamp(total_length)} -> {microseconds_to_audio_timestamp(total_length + l)}')
            total_length += l
            prediction_text = process_sample(d['audio'], processor, model)
            wer = 100 * metric.compute(predictions=[prediction_text], references=[transcription])
            total_wer += wer
            average_wer = total_wer/(i+1)
            print(prediction_text)
            print(transcription)
            print(f'current WER={wer}; total WER={total_wer}; average WER={average_wer}')

        exit_values.append(f'Model {m}. File {dir_name.name}. Total WER={total_wer}; average WER={average_wer}')

print(exit_values)
