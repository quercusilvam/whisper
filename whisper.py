#!/opt/conda/envs/pytorch/bin/python
import argparse
# import evaluate
import os
import shutil
import torch

from datasets import Dataset
from pathlib import Path
from whisperAWSHelper import WhisperAWSHelper
from whisperProgressLogger import WhisperProgressLogger
from whisperplus import (
    SpeechToTextPipeline,
    ASRDiarizationPipeline,
    format_speech_to_dialogue,
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


INPUT_DIR = 'audio_chunks/'
TEST_DIR = 'test/'
CHUNKS_DIR = 'chunks/'
OUTPUT_DIR = 'output/'
DS_DIR = 'ds/'
MODEL = 'openai/whisper-large-v3' #'nyrahealth/CrisperWhisper'
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
parser.add_argument('-hf', '--hf_token', default=None,
                    help=f'Set the HugginFace Token needed by pyannote/speaker-diarization')
args = parser.parse_args()

aws_bucket_name = args.aws_bucket_name
if aws_bucket_name:
    is_source_aws = True
    wah = WhisperAWSHelper(aws_bucket_name, CRYPT_KEY_FILE)
else:
    is_source_aws = False
    wah = None

is_only_benchmark = args.benchmark
model_name = args.model
if model_name is None:
    model_name = MODEL
hf_token = args.hf_token

def main():
    """Main function."""
    print(f'Starting loading model {model_name}')
    # pipeline = ASRDiarizationPipeline.from_pretrained(
    #     asr_model=model_name,
    #     diarizer_model="pyannote/speaker-diarization@2.1",
    #     use_auth_token=hf_token,
    #     chunk_length_s=30,
    # )
    mstt_pipeline = MySpeechToTextPipeline(model_id=model_name)

    if is_only_benchmark:
        benchmark_model(mstt_pipeline)
    else:
        generate_transcript(mstt_pipeline)


def benchmark_model(processor, model):
    """Calculate total WER for all test_ds for test_model."""
    print(f'Benchmarking model {model_name}')
    exit_values = []
    # metric = evaluate.load('wer')
    metric = ''

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
        prediction_text.append(process_sample(d['file'], test_processor, test_model))
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


def generate_transcript(mstt_pipeline):
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
        output_file_name = os.path.join(OUTPUT_DIR, f'{ds_name}.{TRANSCRIPTION_FILE_FORMAT}')
        if last_processed_chunk >= 0:
            print(f'Resume processing dataset "{ds_name}" after chunk {last_processed_chunk}')
            download_partial_transcription(output_file_name)
        else:
            print(f'Start transcription of dataset "{ds_name}"')

        ds = load_dataset(dir_path)

        with open(output_file_name, 'a', encoding='utf-8') as file:
            for i, d in enumerate(ds):
                if i <= last_processed_chunk:
                    print('Chunk already processed. Skip')
                    continue
                else:
                    print('Chunk already processed. Skip')
                    l = d['len']
                    current_length += l
                    prediction_text = process_sample(str(dir_path)+'/'+d['filename'], mstt_pipeline)
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
    print(result)
    return result


def upload_file_to_aws(file_name, temp_dir=None):
    """Upload file to AWS if it is used."""
    if temp_dir is None:
        temp_dir = Path(file_name).parent
    if is_source_aws:
        wah.upload_to_aws(file_name, temp_dir)


def load_dataset(dir_path):
    """Load dataset from local storage. If needed download it from AWS."""
    if is_source_aws:
        wah.download_from_aws(dir_path)
    return Dataset.load_from_disk(str(dir_path))


def download_partial_transcription(file_path):
    """If transcription is in AWS, download already created file and add new text to it."""
    if is_source_aws:
        wah.download_from_aws(file_path)


def init_output_dir(dir_path):
    """Create (if not exists) chunk & ds output dirs."""
    os.makedirs(dir_path, exist_ok=True)


def microseconds_to_audio_timestamp(microseconds: int):
    """Convert microseconds (int) to string like 00:00:00.0000"""
    hours = microseconds // 3600000
    microseconds -= hours * 3600000
    minutes = microseconds // 60000
    microseconds -= minutes * 60000
    seconds = microseconds // 1000
    microseconds -= seconds * 1000
    milliseconds = int(microseconds / 100)
    return '{:02d}:{:02d}:{:02d}.{:02d}'.format(hours, minutes, seconds, milliseconds)


def process_sample(sample, mstt_pipeline):
    """Process audio sample, generate and return text."""
    return mstt_pipeline(sample, language='pl', return_timestamps=True)


class MySpeechToTextPipeline(SpeechToTextPipeline):
    """Wrapper for SpeechToTextPipeline that returns timestamp as well"""

    def __call__(self,
                 audio_path: str,
                 model_id: str = 'openai/whisper-large-v3',
                 language: str = 'turkish',
                 return_timestamps: bool = False):
        """
        Converts audio to text using the pre-trained speech recognition model.

        Args:
            audio_path (str): Path to the audio file to be transcribed.
            model_id (str): Identifier of the pre-trained model to be used for transcription.

        Returns:
            str: Transcribed text from the audio.
        """
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            'automatic-speech-recognition',
            model=self.model,
            torch_dtype=torch.float16,
            chunk_length_s=30,
            max_new_tokens=128,
            batch_size=24,
            return_timestamps=return_timestamps,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            model_kwargs={'use_flash_attention_2': True},
            generate_kwargs={'language': language},
        )

        result = ''
        if return_timestamps:
            for c in pipe(audio_path)['chunks']:
                if c['timestamp'][0] is None:
                    result += '[??:??:??.??] '
                else:
                    result += '[' + microseconds_to_audio_timestamp(int(c['timestamp'][0]) * 1000) + '] '
                result += c['text'] + '\n'
        else:
            result = pipe(audio_path)['text']

        return result


main()
