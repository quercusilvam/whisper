#!/usr/bin/env python3
import argparse
import os
import evaluate
import gc
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio

INPUT_DIR = 'audio_chunks/'
TEST_DIR = 'test/'
CHUNKS_DIR = 'chunks/'
DS_DIR = 'ds/'
MODEL = 'openai/whisper-small' # 'bardsai/whisper-small-pl']


def main():
    """Main function."""
    processor = WhisperProcessor.from_pretrained(MODEL, language='pl', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.config.forced_decoder_ids = None

    if is_only_benchmark:
        benchmark_model(processor, model)
    else:
        generate_transcript(processor, model)


def benchmark_model(processor, model):
    """Calculate total WER for all test_ds for test_model."""
    print(f'Benchmarking model {MODEL}')
    exit_values = []
    metric = evaluate.load('wer')

    for dir_name in os.scandir(os.path.join(TEST_DIR, DS_DIR)):
        print(f'Read dataset {dir_name.name}')
        ds = Dataset.load_from_disk(dir_name.path)

        calculated_wer = calculate_wer(processor, model, ds, metric)

        t = 'Model {}. File {}. Total WER={:.2f}; average WER={:.2f}'
        exit_values.append(t.format(MODEL, dir_name.name, calculated_wer['total_wer'], calculated_wer['average_wer']))

    print(exit_values)


def calculate_wer(test_processor, test_model, test_ds, metric):
    """Calculate WER for given test_processor/test_model based on test_ds."""
    total_wer = 0
    average_wer = 0
    for i, d in enumerate(test_ds):
        l = d['len']
        transcription = d['transcription']
        prediction_text = process_sample(d['audio'], test_processor, test_model)
        wer = 100 * metric.compute(predictions=[prediction_text], references=[transcription])
        total_wer += wer
        average_wer = total_wer / (i + 1)
        t = 'Chunk {}, current WER={:.2f}; total WER={:.2f}; average WER={:.2f}'
        print(t.format(i + 1, wer, total_wer, average_wer))
    return dict({'total_wer': total_wer, 'average_wer': average_wer})


def generate_transcript(processor, model):
    """Generate transcript for input datasets."""
    print(f'Generating transcript based on model {MODEL}')
    for dir_name in os.scandir(os.path.join(TEST_DIR, DS_DIR)):
        print(f'Read dataset {dir_name.name}')
        ds = Dataset.load_from_disk(dir_name.path)

        total_length = 0
        for i, d in enumerate(ds):
            l = d['len']
            print(f'{microseconds_to_audio_timestamp(total_length)} -> {microseconds_to_audio_timestamp(total_length + l)}')
            total_length += l
            prediction_text = process_sample(d['audio'], processor, model)
            print(prediction_text)


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


parser = argparse.ArgumentParser(description='Transcript input dataset with audio data to text. Also benchmark models')
parser.add_argument('-b', '--benchmark', action='store_true',
                    help='If set the script will only benchmark model(s) based on test dataset(s)')
args = parser.parse_args()

is_only_benchmark = args.benchmark

main()
