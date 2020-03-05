import os
from pkg_resources import working_set, Requirement
import sys

def get_cache_dir():
    # path = os.path.abspath(sys.modules[KB.__module__].__file__)
    distrib = working_set.find(Requirement.parse('zincbase'))
    path = distrib.location + '/zincbase/.cache/'
    # path = '/'.join(path.split('/')[:-1]) + '/.cache/'
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def check_file_exists(filename):
    return os.path.isfile(filename)