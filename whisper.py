#!/usr/bin/env python3
import argparse
import os
import evaluate
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio

INPUT_DIR = 'audio_chunks/'
TEST_DIR = 'test/'
CHUNKS_DIR = 'chunks/'
OUTPUT_DIR = 'output/'
DS_DIR = 'ds/'
MODEL = 'openai/whisper-large-v2'

TRANSCRIPTION_FILE_FORMAT = 'txt'

parser = argparse.ArgumentParser(description='Transcript input dataset with audio data to text. Also benchmark models')
parser.add_argument('-b', '--benchmark', action='store_true',
                    help='If set the script will only benchmark model(s) based on test dataset(s)')
parser.add_argument('-m', '--model', default=MODEL,
                    help='Set the custom model to use')
args = parser.parse_args()

is_only_benchmark = args.benchmark
model_name = args.model


def main():
    """Main function."""
    print(f'Starting loading model {model_name}')
    if torch.cuda.is_available():
        processor = WhisperProcessor.from_pretrained(model_name, language='pl', task='transcribe')
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to('cuda')
    else:
        processor = WhisperProcessor.from_pretrained(model_name, language='pl', task='transcribe')
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = None
    print('Ended loading model')

    if is_only_benchmark:
        benchmark_model(processor, model)
    else:
        generate_transcript(processor, model)


def benchmark_model(processor, model):
    """Calculate total WER for all test_ds for test_model."""
    print(f'Benchmarking model {model_name}')
    exit_values = []
    metric = evaluate.load('wer')

    for dir_name in os.scandir(os.path.join(TEST_DIR, DS_DIR)):
        print(f'Read dataset {dir_name.name}')
        ds = Dataset.load_from_disk(dir_name.path)

        calculated_wer = calculate_wer(processor, model, ds, metric)

        t = 'Model {}. File {}. WER={:.2f}'
        exit_values.append(t.format(model_name, dir_name.name, calculated_wer['wer']))

    print(exit_values)


def calculate_wer(test_processor, test_model, test_ds, metric):
    """Calculate WER for given test_processor/test_model based on test_ds."""
    transcription = []
    prediction_text = []
    for d in test_ds:
        chunk = d['file']
        print(f'  Read chunk {chunk}')
        transcription.append(d['transcription'])
        prediction_text.append(process_sample(d['audio'], test_processor, test_model))
    pr = ''
    for p in prediction_text:
        pr += ' ' + p
    tr = ''
    for t in transcription:
        tr += ' ' + t
    wer = 100 * metric.compute(predictions=[pr], references=[tr])
    print(pr)
    print('WER={:.2f}'.format(wer))
    return dict({'wer': wer})


def generate_transcript(processor, model):
    """Generate transcript for input datasets."""
    print(f'Generating transcript based on model {model_name}')
    init_output_dir(OUTPUT_DIR)
    for dir_name in os.scandir(os.path.join(INPUT_DIR, DS_DIR)):
        print(f'Read dataset {dir_name.name}')
        ds = Dataset.load_from_disk(dir_name.path)

        total_length = 0
        output_text=[]
        output_file_name = os.path.join(OUTPUT_DIR, f'{dir_name.name}.{TRANSCRIPTION_FILE_FORMAT}')
        with open(output_file_name, 'w') as file:
            for i, d in enumerate(ds):
                l = d['len']
                time_mark = f'{microseconds_to_audio_timestamp(total_length)} -> {microseconds_to_audio_timestamp(total_length + l)}'
                print(time_mark)
                file.write(time_mark + '\n')
                total_length += l
                prediction_text = process_sample(d['audio'], processor, model)
                file.write(prediction_text + '\n')
                file.flush()


def init_output_dir(dir_path):
    """Create (if not exists) chunk & ds output dirs."""
    os.makedirs(dir_path, exist_ok=True)


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
    if torch.cuda.is_available():
        input_features = input_features.to('cuda')

    predicted_ids = model.generate(input_features,
                                   language='pl',
                                   is_multilingual=True,
                                   task='transcribe',
                                   forced_decoder_ids=forced_decoder_ids)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return text[0]


main()
