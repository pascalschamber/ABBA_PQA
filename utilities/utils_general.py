import os
from pathlib import Path
import timeit
import re


def verify_outputdir(adir):
    # make an outputdir if it doesn't exists
    if not os.path.exists(adir):
        os.mkdir(adir)
    return adir

def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def time_this(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

def clean_filename(filename, replace_with='', replace_spaces=None):
    # remove any character not allowed in Windows filenames, Note: will interfere if passed a file path
    # This includes foreign characters and special symbols
    # optionally replace spaces with a character
    cleaned_filename = re.sub(r'[\\/:*?"<>|\[\]{}\n\',]', replace_with, filename)

    # Remove trailing spaces or periods
    cleaned_filename = re.sub(r'[ \.]+$', '', cleaned_filename)
    # replace spaces
    if replace_spaces is not None:
        cleaned_filename = cleaned_filename.replace(' ', replace_spaces)

    return cleaned_filename