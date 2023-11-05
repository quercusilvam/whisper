#!/usr/bin/env python3

import json
import os


class WhisperProgressLogger():
    """Log progress of converting audio to text.

    Allow resuming transcription from saved moment, not from beginning.
    Help manage spot instances on AWS saving progress and transcription parts to S3, resuming from last position,
    skip transcription of the same datasets
    """
    _progress_filename = 'progress.json'
    _json_datasets_label = 'datasets'
    _json_ds_name_label = 'ds_name'
    _json_last_processed_chunk_label = 'last_processed_chunk'
    _json_last_processed_chunk_new_entry = -1
    _json_total_processed_length_label = 'total_processed_length'
    _json_total_processed_length_new_entry = 0
    _json_ds_processed_label = 'processed'

    def __init__(self, output_dir_path, is_source_aws=False):
        """Set output dir path. If source is AWS, download files from there"""
        self.output_dir_path = output_dir_path
        self.progress_file_path = os.path.join(output_dir_path, self._progress_filename)
        self.progress_file_data = {}
        self.progress_current_ds_data = {}
        self.is_source_aws = is_source_aws
        self._read_progress_file()

    def _read_progress_file(self):
        """Read content of the progress file into memory."""
        self._download_file_from_aws()
        try:
            with open(self.progress_file_path, 'r') as file:
                self.progress_file_data = json.load(file)
        except FileNotFoundError:
            print('No progress file. Set empty one')
            self.progress_file_data = {self._json_datasets_label: []}
            self._write_progress_file()

    def _write_progress_file(self):
        """Write updated data to progress file"""
        with open(self.progress_file_path, 'w') as file:
            file.writelines(json.dumps(self.progress_file_data, sort_keys=False, indent=1))

    def _download_file_from_aws(self):
        """Download progress file from AWS (if set)"""
        if self.is_source_aws:
            pass

    def _upload_file_to_aws(self):
        """Upload progress file to AWS (if set)"""
        if self.is_source_aws:
            pass

    def check_dataset_progress(self, dataset_name):
        """Check progress of given dataset in progress file."""
        fresh_ds_entry = {self._json_ds_name_label: dataset_name,
                          self._json_ds_processed_label: False,
                          self._json_last_processed_chunk_label: self._json_last_processed_chunk_new_entry,
                          self._json_total_processed_length_label: self._json_total_processed_length_new_entry}

        datasets = self.progress_file_data[self._json_datasets_label]
        for d in datasets:
            if dataset_name == d[self._json_ds_name_label]:
                self.progress_current_ds_data = d
                return

        datasets.append(fresh_ds_entry)
        self.progress_current_ds_data = datasets[-1]
        self._write_progress_file()

    def is_dataset_processed(self):
        """Return true if dataset processed."""
        return self.progress_current_ds_data[self._json_ds_processed_label]

    def get_dataset_last_processed_chunk(self):
        """Return number of last chunk processed."""
        return self.progress_current_ds_data[self._json_last_processed_chunk_label]

    def get_dataset_total_processed_length(self):
        """Return number of total processed audio length."""
        return self.progress_current_ds_data[self._json_total_processed_length_label]

    def update_dataset_progress(self, processed_chunk_number, total_processed_length):
        """Update the number or processed chunks for dataset progress"""
        self.progress_current_ds_data[self._json_last_processed_chunk_label] = processed_chunk_number
        self.progress_current_ds_data[self._json_total_processed_length_label] = total_processed_length
        self._write_progress_file()

    def mark_dataset_as_processed(self):
        """Mark the dataset as processed."""
        self.progress_current_ds_data[self._json_ds_processed_label] = True
        self._write_progress_file()
