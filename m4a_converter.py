#!/usr/target/env python3

"""
Convert .m4a files to .mp3 file
Based on https://github.com/IsaacChapman/python-convert-m4a-to-mp3
"""

import logging
import sys
import glob
import mutagen
import os
from pydub import AudioSegment

# Variables (set for your environment)
music_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'input')
log_file = '/tmp/convert-m4a-to-mp3.log'

# Logging
# Because I want a full log file to potentially reference, but still want to see activity...
# use multiple logging handlers and use ANSI escape code to clear stdout to show a single line
# See https://en.wikipedia.org/wiki/ANSI_escape_code
# If the logging output is longer then the terminal width, it won't be as clean :(

my_logger = logging.getLogger()
my_logger.setLevel(logging.INFO) # Switch from 'INFO' to 'DEBUG' for more detailed output

# Handler for logging to file
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
file_handler.setFormatter(file_formatter)

# Handler for showing only a single log line in stdout
stdout_handler_temporary = logging.StreamHandler(sys.stdout)
# ANSI escape codes:
# '\x1b[nD' cursor back 'n' spaces
# '\x1b[K' clear from cursor to end of the line
stdout_formatter_temporary = logging.Formatter('\x1b[200D\x1b[K%(levelname)s: %(message)s')
stdout_handler_temporary.setFormatter(stdout_formatter_temporary)
stdout_handler_temporary.terminator = '\r' # Should also set the cursor to the beginning of the line

# Handler for typical display of stdout lines
stdout_handler_permanent = logging.StreamHandler(sys.stdout)
stdout_formatter_permanent = logging.Formatter('%(levelname)s: %(message)s')
stdout_handler_permanent.setFormatter(stdout_formatter_permanent)

def switch_stdout_logger(temp_or_perm):
    # Need to clear old handlers
    for handler in my_logger.handlers[:]:
        my_logger.removeHandler(handler)
    # re-add appropriate loggers
    my_logger.addHandler(file_handler)
    if temp_or_perm == 'temp':
        my_logger.addHandler(stdout_handler_temporary)
    elif temp_or_perm == 'perm':
        my_logger.addHandler(stdout_handler_permanent)

# Get list of tags to update (they have to be supported by both mp3 and m4a/mp4 metadata)
from mutagen.easymp4 import EasyMP4Tags
from mutagen.easyid3 import EasyID3
allowed_mp3_tags = list(EasyID3.Set.keys())
allowed_m4a_tags = list(EasyMP4Tags.Set.keys())
tags_to_check = list(set.intersection(set(allowed_mp3_tags), set(allowed_m4a_tags)))

# Show the first message and then switch to showing only a single line
switch_stdout_logger('perm')
my_logger.info('Starting conversion scan of: %s' % music_dir)
switch_stdout_logger('temp')

def short_file_name(file_name):
    return file_name.lstrip(music_dir)

# Iterate through .m4a files in 'music_dir'
os.chdir(music_dir)
for m4a_file in glob.glob(music_dir + '/**/*.m4a', recursive=True):
    my_logger.debug(r'm4a file: %s', short_file_name(m4a_file))
    mp3_file = m4a_file[:-4] + '.mp3'
    if os.path.exists(mp3_file):
        my_logger.info(r'mp3 file already exists: %s', short_file_name(mp3_file))
    else:
        try:
            # Create mp3 file
            my_logger.info(r'Creating mp3 file: %s' % short_file_name(mp3_file))
            AudioSegment.from_file(m4a_file).export(mp3_file, format='mp3')

            # Copy appropiate tags from m4a file to mp3 file
            my_logger.debug(r'Reading file tags: %s' % short_file_name(m4a_file))
            m4a_tags = mutagen.File(m4a_file, easy=True)
            mp3_tags = mutagen.File(mp3_file, easy=True)
            for tag in tags_to_check:
                if tag in m4a_tags:
                    my_logger.debug('writing tag %s: %s' % (tag, m4a_tags[tag]))
                    mp3_tags[tag] = m4a_tags[tag]
            mp3_tags.save()
        except Exception:
            my_logger.exception('PROBLEM WITH CONVERTING: %s' % (short_file_name(m4a_file)))
            continue

# Reset stdout formatter to show the final message
switch_stdout_logger('perm')
my_logger.info('Finished conversion scan of: %s' % music_dir)
