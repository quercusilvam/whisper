#!/usr/bin/env python3

import os
import boto3

from pathlib import Path
from whisperCryptoHelper import WhisperCryptoHelper


class WhisperAWSHelper:
    """This class handle AWS download/upload of Datasets, transcriptions, & progress file."""

    _s3 = boto3.resource('s3')

    def __init__(self, bucket_name, key_file_path):
        """Setup bucket name, local dst/src dirs, key_file_path."""
        self.bucket_name = bucket_name
        self.aws_bucket = self._s3.Bucket(bucket_name)
        self.wch = WhisperCryptoHelper(key_file_path)

    def check_if_object_exists_in_bucket(self, filepath):
        """Check if given object exists in S3 bucket."""
        exists = False
        for obj in self.aws_bucket.objects.filter(Prefix=filepath):
            if obj.key == filepath:
                exists = True
                break
        if not exists:
            print(f'File {filepath} not found on S3 bucket!')
        return exists

    def get_list_of_available_objects_in_dir(self, dir_path):
        """Get lis of available object on S3 bucket in given dir."""
        result = []
        # Ensure there is '/' at the end of dir path
        dir_path = os.path.join(dir_path, '')
        for obj in self.aws_bucket.objects.filter(Delimiter='/', Prefix=dir_path):
            if obj.key != dir_path:
                result.append(obj.key)
        return result

    def download_from_aws(self, filepath, dest_dir=None):
        """Download selected file from AWS"""
        result = None
        parent = Path(filepath).parent
        if parent != '.':
            stem_fullpath = os.path.join(Path(filepath).parent, Path(filepath).stem)
        else:
            stem_fullpath = Path(filepath).stem
        s3_filepath = stem_fullpath + self.wch.get_zip_crypt_file_extension()

        if self.check_if_object_exists_in_bucket(s3_filepath):
            filename = Path(filepath).name
            if dest_dir:
                crypt_fn = os.path.join(dest_dir, filename)
            else:
                crypt_fn = filename

            self.aws_bucket.download_file(s3_filepath, crypt_fn)
            result = self.wch.decrypt_and_unzip(crypt_fn, dest_dir)

        return result

    def upload_to_aws(self, filepath, tmp_dir=None, dest_s3_dir=None):
        """Upload selected file to AWS"""
        crypt_file = self.wch.encrypt_and_zip(filepath, tmp_dir)
        s3_filename = Path(crypt_file).name
        if dest_s3_dir:
            s3_filepath = os.path.join(dest_s3_dir, s3_filename)
        else:
            parent = Path(filepath).parent
            if os.path.isdir(parent):
                s3_filepath = os.path.join(parent, s3_filename)
            else:
                s3_filepath = s3_filename
        return self.aws_bucket.upload_file(crypt_file, s3_filepath)
