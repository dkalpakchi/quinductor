import os
import codecs

import setuptools

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name='quinductor',
    version=get_version("src/quinductor/__init__.py"),
    description='A package implementing a multi-lingual question generation method described in https://arxiv.org/abs/2103.10121',
    url='https://github.com/dkalpakchi/quinductor',
    author='Dmytro Kalpakchi',
    install_requires=[
        'numpy',
        'udon2',
        'pypeg2',
        'stanza',
        'tqdm',
        'dill',
        'jsonlines'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    package_dir={"": "src"},
    package_data={
        'quinductor': ['lang_spec_feats.json'],
    },
    packages=setuptools.find_packages(where="src")
)
