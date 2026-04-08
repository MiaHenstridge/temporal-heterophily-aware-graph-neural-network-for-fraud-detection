from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ext_modules = [
    Pybind11Extension("sampler_core", 
                      ['src/sampler_core.cpp'],
                      extra_compile_args = ['-fopenmp'],
                      extra_link_args = ['-fopenmp'],),
]

setup(
    name = "sampler_core",
    version = "0.0.1",
    author = "Hongkuan Zhou",
    author_email = "hongkuaz@usc.edu",
    url = "https://tedzhouhk.github.io/about/",
    description = "Parallel Sampling for Temporal Graphs",
    ext_modules = ext_modules,
)