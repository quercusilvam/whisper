#!/usr/bin/env python3

import os
import zipfile

from cryptography.fernet import Fernet

from pathlib import Path


class WhisperCryptoHelper:
    """Zip and encrypt given files or folders"""
    _crypt_file_extension = '.crypt'
    _zip_file_extension = '.zip'

    def __init__(self, key_file_path):
        """Use given key file to encrypt/decrypt data.

        Create new key_file_path if given one is missing."""
        try:
            key = open(key_file_path, 'rb').read()
        except FileNotFoundError:
            key = self._generate_new_key(key_file_path)
        self.fernet = Fernet(key)

    def _generate_new_key(self, key_file_path):
        """Generate new key and save it to given file."""
        key = Fernet.generate_key()
        with open(key_file_path, 'wb') as key_file:
            key_file.write(key)
        return key

    def _zip(self, path, zipname=None, dest_dir=None):
        """Create zip from given path and place it in directory.

        :param: path Path to file or directory, that should be zipped.
        :param: zipname If provided, it will be used. If not - zip file will be named after file/dir from path
        :param: dest_dir Custom destination directory
        :return: path to zipfile
        """
        if zipname is None:
            if os.path.isfile(path):
                zipname = Path(path).stem + self._zip_file_extension
            else:
                zipname = Path(path).name + self._zip_file_extension
        if dest_dir:
            zipname = os.path.join(dest_dir, Path(zipname).name)

        with zipfile.ZipFile(zipname, 'w') as zip_ref:
            if os.path.isfile(path):
                zip_ref.write(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        zip_ref.write(os.path.join(root, file),
                                      os.path.relpath(os.path.join(root, file),
                                                      os.path.join(path, '..')))

        return zipname

    def _unzip(self, filepath, dest_dir=None):
        """Unzip given file."""
        result = []
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            if dest_dir:
                for n in zip_ref.namelist():
                    result.append(os.path.join(dest_dir, n))
            else:
                result = zip_ref.namelist()
        return result

    def encrypt_and_zip(self, path, dst_dir=None):
        """Zip and encrypt given file/dir with provided key."""
        zippath = self._zip(path, dest_dir=dst_dir)
        crypt_filepath = zippath + self._crypt_file_extension
        with open(zippath, 'rb') as file:
            encrypted_data = self.fernet.encrypt(file.read())
        with open(crypt_filepath, 'wb') as file:
            file.write(encrypted_data)

        Path(zippath).unlink()
        return crypt_filepath

    def decrypt_and_unzip(self, filepath, dst_dir=None):
        """Decrypt the given file with provided key, then unzip"""
        decrypt_filepath = Path(filepath).stem
        with open(filepath, 'rb') as file:
            encrypted_data = file.read()
        decrypted_data = self.fernet.decrypt(encrypted_data)
        with open(decrypt_filepath, 'wb') as file:
            file.write(decrypted_data)

        files = self._unzip(decrypt_filepath, dest_dir=dst_dir)
        Path(decrypt_filepath).unlink()
        return files

    def add_file_extension(self, filepath):
        """Return filepath with extension added to zipped & encrypted files."""
        return filepath + self._zip_file_extension + self._crypt_file_extension

    def remove_file_extension(self, filepath):
        """Return filepath with extension removed."""
        return os.path.join(Path(filepath).parent, Path(Path(filepath).stem).stem)
