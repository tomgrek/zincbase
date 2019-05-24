import os, sys

from zincbase import KB

def get_cache_dir():
    path = os.path.abspath(sys.modules[KB.__module__].__file__)
    path = '/'.join(path.split('/')[:-1]) + '/.cache/'
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def check_file_exists(filename):
    return os.path.isfile(filename)