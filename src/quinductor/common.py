import os
import csv
import sys
import unicodedata
import re
from collections import defaultdict
from pathlib import Path
from pprint import pprint
import logging

import numpy as np

COORDINATE_CLAUSES = ['conj']
SUBORDINATE_CLAUSES = ['csubj', 'xcomp', 'ccomp', 'advcl', 'acl', 'acl:relcl']

FUNCTION_WORDS = ['DET', 'PART', 'INTJ', 'SYM'] # AUX is a special case

MODIFIER_RELS = ["advmod", "amod"]

PUNCT_TABLE = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

HOME_DIR = str(Path.home())
DEFAULT_TEMPLATES_DIR = os.getenv(
    'QUINDUCTOR_RESOURCES_DIR',
    os.path.join(HOME_DIR, 'quinductor_resources')
)
QUINDUCTOR_RESOURCES_GITHUB = 'https://raw.githubusercontent.com/dkalpakchi/quinductor/master/templates'

MODELS = {
    'ar': {
        'tydiqa': {
            'templates': 1614104416496133
        },
        'pos_ngrams': ['ar_padt_train.txt'],
        'default': 'tydiqa'
    },
    'en': {
        'tydiqa': {
            'templates': 16132054753040054
        },
        'squad': {
            'templates': 16432660346112196
        },
        'pos_ngrams': ['ewt_train_freq.txt', 'ewt_dev_freq.txt'],
        'default': 'tydiqa'
    },
    'fi': {
        'tydiqa': {
            'templates': 16132078825085254
        },
        'pos_ngrams': ['fi_tdt_train.txt'],
        'default': 'tydiqa'
    },
    'id': {
        'tydiqa': {
            'templates': 16140609246000547
        },
        'pos_ngrams': ['id_gsd_train.txt'],
        'default': 'tydiqa'
    },
    'ja': {
        'tydiqa': {
            'templates': 16140572221308537
        },
        'pos_ngrams': ['ja_gsd_train.txt'],
        'default': 'tydiqa'
    },
    'ko': {
        'tydiqa': {
            'templates': 16140582210609627
        },
        'pos_ngrams': ['ko_gsd_train.txt'],
        'default': 'tydiqa'
    },
    'ru': {
        'tydiqa': {
            'templates': 1613204358381249
        },
        'pos_ngrams': ['ru_syntagrus_train.txt'],
        'default': 'tydiqa'
    },
    'te': {
        'tydiqa': {
            'templates': 16140691545631247
        },
        'pos_ngrams': ['te_mtg_train.txt'],
        'default': 'tydiqa'
    }
}


def get_logger():
    logger = logging.getLogger('quinductor')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def get_default_model_path(lang, mtype=None):
    if lang in MODELS:
        model = mtype or MODELS[lang]['default']
        return os.path.join(DEFAULT_TEMPLATES_DIR, lang, model, str(MODELS[lang][model]['templates']))
    else:
        logger = logging.getLogger('quinductor')
        logger.error(
            """The language {} currently has no available models.
            Please create your own model and provide it using script arguments""".format(args.lang)
        )
        sys.exit(1)


def get_default_guards_path(lang):
    return os.path.join(get_default_model_path(lang), 'guards.txt')


def get_default_templates_path(lang):
    return os.path.join(get_default_model_path(lang), 'templates.txt')


def get_default_examples_path(lang):
    return os.path.join(get_default_model_path(lang), 'sentences.txt')


class TemplateElement:
    def __init__(self, root_chain, node, last_span_id=None, is_subtree=False, is_lemma=False):
        self.__chain = root_chain
        self.__id = (node.id if last_span_id is None else last_span_id) if node else None
        self.__node = node
        self.__positives = []
        self.__negatives = []
        self.__is_subtree = is_subtree
        # the subtrees that can't be added definitely because the child of the node is already in the positives list
        self.__taboo = []
        self.__is_lemma = is_lemma

    @property
    def is_lemma(self):
        return self.__is_lemma

    def copy_positives(self, elements):
        for e in elements:
            self.__positives.extend(e.positives)
            self.__taboo.extend(e.taboo)

    def copy_negatives(self, elements):
        for e in elements:
            self.add_negatives(e.negatives)

    def add_positives(self, elements):
        for e in elements:
            self.__positives.append(e)
            if e[-1].endswith('*'):
                e_copy = list(e)
                e_copy[-1] = e_copy[-1][:-1]
                self.__taboo.append(e_copy)

            for k in range(len(e)):
                self.__taboo.append(e[:-1-k])

    def add_negatives(self, elements):
        # print("TABOO", self.__taboo)
        # print("POS", self.__positives)
        for e in elements:
            # print(e)
            # negatives list should be quite small (up to 10 elements),
            # so it should be quick to check `in` relationship
            if e not in self.__positives and e not in self.__negatives and e not in self.__taboo:
                self.__negatives.append(e)
        # print(self.__negatives)

    def merge_negatives(self, children_map):
        # TODO: wrong merging procedure
        # need to check if they can be merged in the first place
        if self.__negatives:
            # TODO: check if any of the values of the children map is fully in negatives
            # if so, swap all those elements for the corresponding key
            # NOTE: key is a str, so split it by '.' first
            set_negatives = set(map(lambda x: ".".join(x), self.__negatives))
            sorted_children = sorted(children_map.items(), reverse=True)
            for k, v in sorted_children:
                set_values = set(map(lambda x: ".".join(x), v))
                if set_values and (set_values & set_negatives) == set_values:
                    for node in v:
                        try:
                            self.__negatives.remove(node)
                        except ValueError:
                            # means we've already deleted this element as a part of other chain
                            pass
                    self.__negatives.append(k.split('.'))

    @property
    def positives(self):
        return self.__positives

    @property
    def negatives(self):
        return self.__negatives

    @property
    def taboo(self):
        return self.__taboo

    @property
    def node(self):
        return self.__node

    @property
    def chain(self):
        return self.__chain

    @property
    def id(self):
        return self.__id

    @property
    def is_subtree(self):
        return self.__is_subtree

    def __len__(self):
        return len(self.__chain)

    def __str__(self):
        prefix = '.'.join([x.replace('*', '') for x in self.__chain])
        inside = f"w{'.' + prefix if self.__chain else ''}"
        negate = " ".join([f"- {'.'.join(n).replace(prefix + '.', '')}" for n in self.__negatives])
        if negate:
            inside += " " + negate
        if self.__is_subtree:
            return f"<{inside}>"
        else:
            return f"[{inside}.lemma]" if self.__is_lemma else f"[{inside}]"

    def __and__(self, other):
        res = []
        for c in self.__chain:
            if c in other.chain:
                res.append(c)
            else:
                # the longest prefix subchain found
                break
        return res


def chain(node1, node2, include_ids=False, only_ids=False):
    """
    Find a chain of dependency tags from `node1` to `node2` (if possible)
    
    :param      node1:  The node 1
    :type       node1:  udon2.Node
    :param      node2:  The node 2
    :type       node2:  udon2.Node
    """
    node, chain = node2, []
    while not node.is_identical(node1, ""):
        chain.append(node.id if only_ids else f"{node.deprel}#{int(node.id)}" if include_ids else node.deprel)
        node = node.parent
    chain.reverse()
    return chain


def chain_to_root(node, include_ids=False, only_ids=False):
    n, chain = node, []
    while not n.is_root():
        chain.append(node.id if only_ids else f"{node.deprel}#{int(node.id)}" if include_ids else node.deprel)
        n = n.parent
    chain.reverse()
    return chain

def get_question_word(question_nodes, lang):
    N = len(question_nodes)
    start_word, next_index, cand = None, 1, question_nodes[0].form.lower()
    if cand in lang.OW_QUESTION_WORDS:
        start_word = cand

    j = 1
    while cand not in lang.MW_QUESTION_WORDS and j < N:
        cand += f" {question_nodes[j].form.lower()}"
        j += 1

    if j < N:
        start_word = cand
        next_index = j
    return start_word, next_index


def get_sentence_by_answer(answer, context, segmenter):
    # segmenter is a Stanza tokenizer
    doc = segmenter(context.strip().replace(u'\xa0', ' '))
    it = 0
    ans = answer['text'].strip().replace(u'\xa0', ' ')
    for s in doc.sentences:
        it += sys.getsizeof(s.text)
        sent = s.text

        if ans in sent and it >= answer['end']:
            for p in sent.strip().split('\n'):
                # TODO: some of the cases will be invalid, because of lists or "blankett nummer" cases
                if ans in p:
                    return p.strip()

def get_statistics(fname):
    qw_stat = defaultdict(int)
    with open(fname) as f:
        for line in f:
            if line.strip():
                question, answer, base_sentence = line.split("|")
                qw_stat[question.split()[0]] += 1
    return qw_stat


def get_difference(a, b):
    """
    Get the difference between `a` and `b` preserving the order of elements in `a`
    """
    if type(a) == str:
        a = a.split()
    if type(b) == str:
        b = b.split()

    diff = []
    for word in a:
        if word not in b:
            diff.append(word)
    return diff


def get_intersection(a, b):
    """
    Get the difference between `a` and `b` preserving the order of elements in `a`
    """
    if type(a) == str:
        a = a.split()
    if type(b) == str:
        b = b.split()

    if not a or not b:
        return []

    same = []
    for word in a:
        if word in b:
            same.append(word)
    return same


def check_multiword_qw(lst, lang):
    return " ".join(lst) in lang.MW_QUESTION_WORDS

def load_pos_ngrams(folder):
    bigrams_freq = defaultdict(lambda: defaultdict(int))
    trigrams_freq = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    # morph_vocab = set()
    for fname in os.listdir(folder):
        definitions = True
        id2tag = {}
        with open(os.path.join(folder, fname)) as f:
            reader = csv.reader(f, delimiter=',')

            for line in reader:
                if line:
                    if definitions:
                        idx = line[0]
                        tag = ",".join(line[1:])
                        if '/' in tag:
                            pos, morph = tag.split('/')
                            morph_tags = sorted(morph.split('|'))
                            # for m in morph_tags:
                            #     morph_vocab.add(m)
                            id2tag[int(idx)] = f"{pos}/{'|'.join(morph_tags)}"
                        else:
                            id2tag[int(idx)] = tag
                    else:
                        if len(line) == 3:
                            w1, w2, freq = map(int, line)
                            bigrams_freq[id2tag[w1]][id2tag[w2]] += freq
                        elif len(line) == 4:
                            w1, w2, w3, freq = map(int, line)
                            trigrams_freq[id2tag[w1]][id2tag[w2]][id2tag[w3]] += freq
                else:
                    definitions = False
    
    unigrams_freq = defaultdict(int)    
    for w1 in bigrams_freq:
        for w2 in bigrams_freq[w1]:
            unigrams_freq[w1] += bigrams_freq[w1][w2]
            unigrams_freq[w2] += bigrams_freq[w1][w2]

    log_unigrams = defaultdict(float)
    log_S = np.log(sum(unigrams_freq.values()))
    for w1 in unigrams_freq:
        log_unigrams[w1] = np.log(unigrams_freq[w1]) - log_S

    log_bigrams = defaultdict(lambda: defaultdict(float))
    for w1 in bigrams_freq:
        log_S = np.log(sum(bigrams_freq[w1].values()))
        for w2 in bigrams_freq[w1]:
            log_bigrams[w1][w2] = np.log(bigrams_freq[w1][w2]) - log_S


    log_trigrams = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    for w1 in trigrams_freq:
        for w2 in trigrams_freq[w1]:
            log_S = np.log(bigrams_freq[w1][w2])
            for w3 in trigrams_freq[w1][w2]:
                log_trigrams[w1][w2][w3] = np.log(trigrams_freq[w1][w2][w3]) - log_S

    return log_unigrams, log_bigrams, log_trigrams #, morph_vocab


def read_csv_eval(eval_fname):
    sentences, hypothesis, answers, written = [], [], [], False
    scores, score_sum, total = [], 0, 0
    with open(eval_fname) as f:
        reader = csv.reader(f, delimiter='|')
        headers = next(reader)
        for row in reader:
            if row:
                score_sum += float(row[4])
                total += 1
                if not written:
                    sentences.append(row[1])
                    hypothesis.append(row[2])
                    answers.append(row[3])
                    scores.append(float(row[4]))
                    written = True
            else:
                written = False
    return {
        'hypothesis': hypothesis,
        'sentences': sentences,
        'answers': answers,
        'scores': scores,
        'threshold': score_sum / total
    }


def repr_pos_morph(w):
    if type(w) == str:
        return w

    if w.feats:
        # str(w.feats) should be sorted
        token = f"{w.upos}/{str(w.feats)}"
    else:
        token = w.upos
    return token

def remove_unicode_punctuation(text):
    return text.translate(PUNCT_TABLE)

def is_punctuation(x):
    return unicodedata.category(x).startswith('P')


def remove_unicode_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn')

def load_idf(fname):
    idf = {}
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            try:
                word, v = line
                idf[word] = float(v)
            except:
                continue
    return idf