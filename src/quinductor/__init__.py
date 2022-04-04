"""
Common utilities for Stanza resources.
"""

import os
import sys
import glob
import requests
from pathlib import Path
import json
import hashlib
import zipfile
import shutil

import dill
from tqdm.auto import tqdm

from .common import (
    DEFAULT_TEMPLATES_DIR, QUINDUCTOR_RESOURCES_GITHUB, MODELS,
    get_logger, get_default_model_path, load_pos_ngrams
)
from .guards import load_guards
from .rules import load_templates, load_template_examples
from .core import generate_questions


__version__ = '0.2.0'

logger = get_logger()


# The functions below (until -- END STANZA --) are a snapshot taken from Stanza
# https://github.com/stanfordnlp/stanza/blob/f91ca215e175d4f7b202259fe789374db7829395/stanza/resources/common.py

def ensure_dir(path):
    """
    Create dir in case it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def get_md5(path):
    """
    Get the MD5 value of a path.
    """
    with open(path, 'rb') as fin:
        data = fin.read()
    return hashlib.md5(data).hexdigest()

def unzip(path, filename):
    """
    Fully unzip a file `filename` that's in a directory `dir`.
    """
    logger.debug(f'Unzip: {path}/{filename}...')
    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        f.extractall(path)

def get_root_from_zipfile(filename):
    """
    Get the root directory from a archived zip file.
    """
    zf = zipfile.ZipFile(filename, "r")
    assert len(zf.filelist) > 0, \
        f"Zip file at f{filename} seems to be corrupted. Please check it."
    return os.path.dirname(zf.filelist[0].filename)

def file_exists(path, md5):
    """
    Check if the file at `path` exists and match the provided md5 value.
    """
    return os.path.exists(path) and get_md5(path) == md5

def assert_file_exists(path, md5=None):
    assert os.path.exists(path), "Could not find file at %s" % path
    if md5:
        file_md5 = get_md5(path)
        assert file_md5 == md5, "md5 for %s is %s, expected %s" % (path, file_md5, md5)

def download_file(url, path, proxies, raise_for_status=False):
    """
    Download a URL into a file as specified by `path`.
    """
    verbose = logger.level in [0, 10, 20]
    r = requests.get(url, stream=True, proxies=proxies)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True, \
            disable=not verbose, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))
    if raise_for_status:
        r.raise_for_status()
    return r.status_code

def request_file(url, path, proxies=None, md5=None, raise_for_status=False):
    """
    A complete wrapper over download_file() that also make sure the directory of
    `path` exists, and that a file matching the md5 value does not exist.
    """
    ensure_dir(Path(path).parent)
    if file_exists(path, md5):
        logger.info(f'File exists: {path}.')
        return
    download_file(url, path, proxies, raise_for_status)
    assert_file_exists(path, md5)

# -- END STANZA --


def download(lang):
    # verify if the model is already downloaded and skip if it is (resume if downloaded partially)
    mtype = None
    if '/' in lang:
        lang, mtype = lang.split('/')

    if lang in MODELS:
        logger.info("Downloading Quinductor templates for {}".format(lang))
        lang_dir = os.path.join(DEFAULT_TEMPLATES_DIR, lang)
        
        idf_fname = 'idf_{}.csv'.format(lang)
        request_file(
            '{}/{}/{}'.format(QUINDUCTOR_RESOURCES_GITHUB, lang, idf_fname),
            os.path.join(lang_dir, idf_fname)
        )

        model = mtype or MODELS[lang]['default']

        for fname in ['atmpl.dill', 'qwstats.dill']:
            request_file(
                '{}/{}/{}/{}'.format(QUINDUCTOR_RESOURCES_GITHUB, lang, model, fname),
                os.path.join(lang_dir, model, fname)
            )
        
        pos_ngrams_dir = os.path.join(lang_dir, 'pos_ngrams')
        for fname in MODELS[lang]['pos_ngrams']:
            request_file(
                '{}/{}/{}/{}'.format(QUINDUCTOR_RESOURCES_GITHUB, lang, 'pos_ngrams', fname),
                os.path.join(pos_ngrams_dir, fname)
            )

        model_dir = os.path.join(lang_dir, model, str(MODELS[lang][model]['templates']))
        for fname in ['guards.txt', 'templates.txt', 'sentences.txt']:
            request_file(
                '{}/{}/{}/{}/{}'.format(QUINDUCTOR_RESOURCES_GITHUB, lang, model, str(MODELS[lang][model]['templates']), fname),
                os.path.join(model_dir, fname)
            )
        logger.info("Finished downloading Quinductor templates (saved to {})".format(lang_dir))
    else:
        logger.warning('Templates for language {} are not available.'.format(lang))


def convert2list(n, v):
    if v is not None and type(v) is not list:
        logger.warning("`{}` should be a list, automatically converting".format(n))
        return [str(v)]
    return v


def use(lang=None, templates_folder=None, guards_files=None, templates_files=None, pos_ng_folder=None, example_files=None,
    qw_stat_file=None, a_stat_file=None):
    if not lang and not templates_folder:
        logger.error("use method requires either `lang` or `templates_folder` to be provided")
        sys.exit(1)

    guards_files = convert2list('guards_files', guards_files)
    templates_files = convert2list('templates_files', templates_files)
    example_files = convert2list('example_files', example_files)

    mtype = None
    if '/' in lang:
        lang, mtype = lang.split('/')
    
    temp_folder = templates_folder or get_default_model_path(lang, mtype)

    print(temp_folder)

    ranking_folder = Path(temp_folder).parent
    ng_folder = pos_ng_folder or os.path.join(ranking_folder.parent, 'pos_ngrams')

    rtl = ['ja', 'te', 'ko']

    return {
        'lang': lang,
        'rtl': lang in rtl,
        'join_symbol': ' ', # to output the tokens, maybe make language-specific in future
        'pos_ngrams': load_pos_ngrams(ng_folder),
        'guards': load_guards(guards_files or glob.glob(os.path.join(temp_folder, 'guards.txt'))),
        'templates': load_templates(templates_files or glob.glob(os.path.join(temp_folder, 'templates.txt'))),
        'examples': load_template_examples(example_files or glob.glob(os.path.join(temp_folder, 'sentences.txt'))),
        'qw_stat': dill.load(open(qw_stat_file or os.path.join(ranking_folder, 'qwstats.dill'), 'rb')),
        'a_stat': dill.load(open(a_stat_file or os.path.join(ranking_folder, 'atmpl.dill'), 'rb'))
    }
    