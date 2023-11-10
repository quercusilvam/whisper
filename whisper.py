#!/usr/bin/env python3
import argparse
import boto3
import evaluate
import json
import os
import shutil
import torch

from datasets import Dataset, Audio
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisperAWSHelper import WhisperAWSHelper
from whisperProgressLogger import WhisperProgressLogger


INPUT_DIR = 'audio_chunks/'
TEST_DIR = 'test/'
CHUNKS_DIR = 'chunks/'
OUTPUT_DIR = 'output/'
DS_DIR = 'ds/'
MODEL = 'openai/whisper-tiny'
TRANSCRIPTION_FILE_FORMAT = 'txt'
CRYPT_KEY_FILE = 'fkey.key'

parser = argparse.ArgumentParser(description='Transcript input dataset with audio data to text. Also benchmark models')
parser.add_argument('-ab', '--aws_bucket_name', default=None,
                    help='''Use AWS bucket as input and output (instead of local dir). You need to setup proper
                    credentials using `aws configure` or allow machine to connect to S3 bucket without credentials.
                    In bucket there should be encrypted .zip files, that contains datasets for each file.
                    Script will download one by time, unpack it and process. Result will be pushed to S3 bucket in
                    folder with the same name as dataset.
                    Inside will be parts of transcription (to save long-processing data).''')
parser.add_argument('-b', '--benchmark', action='store_true',
                    help='If set the script will only benchmark model(s) based on test dataset(s)')
parser.add_argument('-m', '--model', default=MODEL,
                    help=f'Set the custom whishper model to use. Defaults to {MODEL}')
args = parser.parse_args()

aws_bucket_name = args.aws_bucket_name
if aws_bucket_name:
    is_source_aws = True
    wah = WhisperAWSHelper(aws_bucket_name, CRYPT_KEY_FILE)
else:
    is_source_aws = False

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
    dataset_list = get_dataset_list()
    print(dataset_list)
    progress = WhisperProgressLogger(whisperAWSHelper=wah)
    for dir_path in dataset_list:
        dir_path = Path(dir_path)
        ds_name = dir_path.name
        print(f'Read dataset "{ds_name}"')
        progress.check_dataset_progress(ds_name)
        if progress.is_dataset_processed():
            print(f'Dataset "{ds_name}" already processed. Skipping')
            continue

        last_processed_chunk = progress.get_dataset_last_processed_chunk()
        current_length = progress.get_dataset_total_processed_length()
        if last_processed_chunk >= 0:
            print(f'Resume processing dataset "{ds_name}" at chunk {last_processed_chunk}')
        else:
            print(f'Start transcription of dataset "{ds_name}"')

        ds = load_dataset(dir_path)

        output_file_name = os.path.join(OUTPUT_DIR, f'{ds_name}.{TRANSCRIPTION_FILE_FORMAT}')
        with open(output_file_name, 'w') as file:
            for i, d in enumerate(ds):
                if i <= last_processed_chunk:
                    print('Chunk already processed. Skip')
                    continue
                else:
                    l = d['len']
                    time_mark = f'{microseconds_to_audio_timestamp(current_length)} -> {microseconds_to_audio_timestamp(current_length + l)}'
                    print(time_mark)
                    file.write(time_mark + '\n')
                    current_length += l
                    prediction_text = process_sample(d['audio'], processor, model)
                    file.write(prediction_text + '\n')
                    file.flush()
                    upload_file_to_aws(output_file_name)
                    progress.update_dataset_progress(i, current_length)

        progress.mark_dataset_as_processed()
        shutil.rmtree(dir_path)


def get_dataset_list():
    """Return list of datasets based on source - localhost or AWS."""
    result = []
    dir_path = os.path.join(INPUT_DIR, DS_DIR)
    if is_source_aws:
        result = wah.get_list_of_available_objects_in_dir(dir_path)
    else:
        result = os.scandir(dir_path)
    return result


def upload_file_to_aws(file_name):
    """Upload file to AWS if it is used."""
    if is_source_aws:
        wah.upload_to_aws(file_name)


def load_dataset(dir_path):
    """Load dataset from local storage. If needed download it from AWS."""
    if is_source_aws:
        wah.download_from_aws(dir_path)
    return Dataset.load_from_disk(dir_path)


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
