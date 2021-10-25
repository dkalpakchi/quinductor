import os
import sys
import copy
import re
import json
import string
import glob
import csv
from operator import itemgetter
from collections import defaultdict
from itertools import product
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

import argparse

import dill
import numpy as np
import stanza
from stanza.utils.conll import CoNLL

from tqdm import tqdm

from quinductor.rules import *
from quinductor.core import *
from quinductor.common import *
from quinductor.guards import load_guards
from quinductor.loaders import *
from quinductor.repro import *

import udon2

np.seterr('raise')


logger = get_logger()


SURVEY_TEMPLATES = {
    'sv': "Meningen: {0}<br>Frågan: {1}<br>Det föreslagna svaret: {2}",
    'en': "Sentence: {0}<br>Question: {1}<br>Suggested answer: {2}",
    'ru': "Предложение: {0}<br>Вопрос: {1}<br>Предложенный ответ: {2}",
    'fi': "Lause: {0}<br>Kysymys: {1}<br>Ehdotettu vastaus: {2}"
}

PREFIXES = {
    'en': 'Disagree',
    'sv': 'Håller inte med',
    'ru': "Не согласен",
    'fi': 'Olen eri mieltä'
}

SUFFIXES = {
    'fi': 'Olen samaa mieltä',
    'ru': "Согласен",
    'sv': "Håller med",
    'en': "Agree"
}

SURVEY_ITEMS = {
    'en': json.load(open(os.path.join('eval', 'en_items.json'))),
    'sv': json.load(open(os.path.join('eval', 'sv_items.json'))),
    'ru': json.load(open(os.path.join('eval', 'ru_items.json'))),
    'fi': json.load(open(os.path.join('eval', 'fi_items.json')))
}

TT_ANSWER_SET_FORMAT = {
    "type": "radio",
    "name": "",
    "choices": list(range(1, 5)),
    "prefix": "",
    "suffix": ""
}


TT_SURVEY_ITEM_FORMAT = {
    "question": "",
    "required": True,
    "order": -1,
    "extra": {},
    "answer_sets": []
}

TT_SURVEY_FORMAT = {
    "name": "Evaluation of reading comprehension questions",
    "items": [],
    "gold": []
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_correct_answer(choices):
    return list(filter(lambda x: x['type'] == 'Correct answer', choices))[0]

def translate(qw):
    return {
        'what': 'vad',
        'which': 'vilken',
        'when': 'när',
        'why': 'varför',
        'how': 'hur',
        "where": 'var',
        "who": 'vem',
        'whose': 'vems'
    }[qw.lower()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='A language for template generation (en, sv are currently supported)')
    parser.add_argument('-f', '--files', type=str, help='Comma-separated list of files to generate questions from')
    parser.add_argument('-t', '--templates-folder', type=str, help="A folder with guards, templates and template_examples subfolders")
    parser.add_argument('-r', '--ranking-folder', type=str, help='A folder with qwstats.dill and atmpl.dill for the language')
    parser.add_argument('-ft', '--format', type=str, help='Data format (tt for Textinator or squad for Squad)')
    parser.add_argument('-pg', '--pos-ngrams', type=str, help='Folder with POS-ngrams for the language only')
    parser.add_argument('-cf', '--case-folding', action='store_true')
    parser.add_argument('-rp', '--remove_punctuation', action='store_true')
    parser.add_argument('-rd', '--remove_diacritics', action='store_true')
    parser.add_argument('-st', '--strict', action='store_true')
    parser.add_argument('-k', '--max-examples', type=int, default=-1, help='Max number of example sentences to be evaluated')
    parser.add_argument('-rtl', '--right-to-left', action='store_true')
    parser.add_argument('-mdb', '--modeldb-url', type=str, default='')
    parser.add_argument('-j', '--join-symbol', type=str, default=' ')
    parser.add_argument('--include-gold', action='store_true')
    args = parser.parse_args()

    if args.modeldb_url:
        client = repro.get_client(args.modeldb_url)
        client.set_project("TyDiQA QG")
        client.set_experiment(args.lang)
        if args.templates_folder[-1] == os.sep:
            timestamp = args.templates_folder[:-1].split(os.sep)[-1]
        else:
            timestamp = args.templates_folder.split(os.sep)[-1]
        run = client.set_experiment_run("{}_{}".format(args.lang, timestamp))
        run.log_hyperparameters({
            'eval_template_folder': args.templates_folder,
            'eval_files': args.files,
            'eval_case_folding': args.case_folding,
            'eval_remove_punct': args.remove_punctuation,
            'eval_right_to_left': args.right_to_left,
            'eval_join_char': args.join_symbol,
            'eval_remove_diacritics': args.remove_diacritics
        }, overwrite=True)

    if args.templates_folder:
        eval_folder = os.path.join(args.templates_folder, 'eval')
    else:
        args.templates_folder = get_default_model_path(args.lang)
        if not os.path.exists(args.templates_folder):
            logger.error(
                """No valid model found. Try downloading by running `quinductor.download({})`
                or providing your own by using script arguments""".format(args.lang)
            )
        eval_folder = 'evaluation'

    if not args.ranking_folder:
        args.ranking_folder = Path(args.templates_folder).parent
    
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    # lm = arpa.loadf(args.language_model)[0]
    # arabic, finnish - include mwt
    # russian - exclude mwt
    dep_proc = 'tokenize,lemma,mwt,pos,depparse' if args.lang in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
    proc = 'tokenize,mwt,pos' if args.lang in ['fi', 'ar'] else 'tokenize,pos'
    stanza_dep_pipe = stanza.Pipeline(lang=args.lang, processors=dep_proc)
    stanza_pipe = stanza.Pipeline(lang=args.lang, processors=proc)
    
    if not args.pos_ngrams:
        args.pos_ngrams = os.path.join(args.ranking_folder, 'pos_ngrams')
    log_prob = load_pos_ngrams(args.pos_ngrams)

    qw_stat = dill.load(open(os.path.join(args.ranking_folder, 'qwstats.dill'), 'rb'))
    a_tmpl = dill.load(open(os.path.join(args.ranking_folder, 'atmpl.dill'), 'rb'))

    if args.format == 'tt':
        data_loader = TextinatorLoader
    elif args.format == 'squad':
        data_loader = SquadLoader
    elif args.format == 'tydiqa':
        data_loader = TyDiQaLoader

    data_file = os.path.join(eval_folder, 'data.dill')
    if os.path.exists(data_file):
        data = dill.load(open(data_file, 'rb'))
    else:
        data = defaultdict(lambda: defaultdict(list))
        for q, a, c in data_loader.from_files(args.files.split(','), args.lang):
            s = get_sentence_by_answer(a, c, stanza_pipe)
            if s:
                # sometimes segmenter can be wrong
                if args.case_folding:
                    s = s.lower()
                    q = q.lower()
                    a['text'] = a['text'].lower()
                if args.remove_punctuation:
                    s = remove_unicode_punctuation(s).strip()
                    q = remove_unicode_punctuation(q).strip()
                    a['text'] = remove_unicode_punctuation(a['text']).strip()
                data[s][q].append(a['text'])
        dill.dump(data, open(data_file, 'wb'))

    fname = "test.conll"
    
    guards_root = load_guards(glob.glob(os.path.join(args.templates_folder, 'guards.txt')))
    templates = load_templates(glob.glob(os.path.join(args.templates_folder, 'templates.txt')))
    template_examples = load_template_examples(glob.glob(os.path.join(args.templates_folder, 'sentences.txt')))

    eval_file = open(os.path.join(eval_folder, "eval_{}.csv".format(args.lang)), 'w')
    writer = csv.writer(eval_file, delimiter='|')
    writer.writerow(['Template', 'Original sentence', 'Generated question', 'Generated answer', 'Score', 'Qw frequency'])

    total_gen, correct_q, correct_t = 0, 0, 0
    total_non_pronoun, total, possible = 0, 0, 0

    ground_truth, hypotheses, hyp_scores, all_scores, survey = [], [], [], [], dict(TT_SURVEY_FORMAT)
    for sent in tqdm(data):
        q_dict = data[sent]

        if not q_dict:
            continue

        if args.max_examples > 0:
            # sample one gold question and answer
            q_dict_keys = list(q_dict.keys())
            ind = np.random.choice(range(len(q_dict_keys)))
            gold_q = q_dict_keys[ind]
            ind_a = np.random.choice(range(len(q_dict[gold_q])))
            gold_a = q_dict[gold_q][ind_a]

        total += len(q_dict)

        sent = re.sub(r' {2,}', '', sent)

        stanza_sent = stanza_dep_pipe(sent)
        with open(fname, 'w') as f:
            conll_list = CoNLL.convert_dict(stanza_sent.to_dict())
            f.write(CoNLL.conll_as_string(conll_list))
        trees = udon2.ConllReader.read_file(fname)

        res = overgenerate_questions(trees, guards_root, templates, template_examples, return_first=False)

        if res:
            idx_sorted_by_scores, qwf, atf, scores = rank(
                res, stanza_pipe, stanza_dep_pipe, qw_stat, a_tmpl, log_prob,
                rtl=args.right_to_left, join_char=args.join_symbol)

            generated, num_recorded_questions = defaultdict(list), 0
            for i in idx_sorted_by_scores:
                if len(res[i]['answer']) == 1 and atf[i] < 1:
                    # if the answer is one word and its sequence of pos-morph tags not appeared in the corpus
                    continue

                if qwf[i] == 0:
                    # if the combination of the question word and the root token of the answer never appeared in the corpus
                    continue

                q = args.join_symbol.join(res[i]['question'])
                a = args.join_symbol.join(res[i]['answer'])

                if args.remove_diacritics:
                    q, a = remove_unicode_diacritics(q), remove_unicode_diacritics(a)

                if num_recorded_questions == 0:
                    # hypotheses and ground truth are to be concatenated by space no matter what,
                    # since nlg-eval splits by space and other packages expect tokenized hypotheses and references
                    gt = []
                    for gt_q in q_dict:
                        q_tokens = [t.text for s in stanza_pipe(gt_q).sentences for t in s.words]
                        if args.remove_diacritics:
                            gt_qq = remove_unicode_diacritics(" ".join(q_tokens))
                        else:
                            gt_qq = " ".join(q_tokens)
                        gt_qq = remove_unicode_punctuation(gt_qq) # since templates are without punctuation
                        gt.append(gt_qq)
                    ground_truth.append(gt)
                    hypotheses.append(q)
                    hyp_scores.append(scores[i])

                    q += '?' # formatting for survey

                    stmpl = SURVEY_TEMPLATES.get(args.lang, SURVEY_TEMPLATES['en'])
                    sitem = copy.deepcopy(TT_SURVEY_ITEM_FORMAT)
                    sitem['extra']['model'] = 'gen'
                    sitem['question'] = stmpl.format(sent.replace('"', '""'), q.replace('"', '""'), a.replace('"', '""'))
                    items = SURVEY_ITEMS.get(args.lang, SURVEY_ITEMS['en'])
                    for item in items:
                        to_add = copy.deepcopy(TT_ANSWER_SET_FORMAT)
                        to_add['name'] = item
                        to_add['prefix'] = PREFIXES.get(args.lang, PREFIXES['en'])
                        to_add['suffix'] = SUFFIXES.get(args.lang, SUFFIXES['en'])
                        sitem['answer_sets'].append(to_add)
                    survey["items"].append(sitem)

                    if args.include_gold:
                        sitem = copy.deepcopy(TT_SURVEY_ITEM_FORMAT)
                        sitem['extra']['model'] = 'gold'
                        sitem['question'] = stmpl.format(sent.replace('"', '""'), gold_q.replace('"', '""'), gold_a.replace('"', '""'))
                        items = SURVEY_ITEMS.get(args.lang, SURVEY_ITEMS['en'])
                        for item in items:
                            to_add = copy.deepcopy(TT_ANSWER_SET_FORMAT)
                            to_add['name'] = item
                            to_add['prefix'] = PREFIXES.get(args.lang, PREFIXES['en'])
                            to_add['suffix'] = SUFFIXES.get(args.lang, SUFFIXES['en'])
                            sitem['answer_sets'].append(to_add)
                        survey["gold"].append(sitem)

                if a.strip() and q.strip():
                    writer.writerow([res[i]['temp_id'], sent, q, a, scores[i], qwf[i]])
                    generated[q].append(a)
                    all_scores.append(scores[i])
                num_recorded_questions += 1

            if idx_sorted_by_scores:
                writer.writerow([])

            # print(correct)
            for q, a_list in generated.items():
                total_gen += len(a_list)

                # print(q, a_list)
                if q in q_dict:
                    correct_q += 1

                    if set(a_list) & set(q_dict[q]):
                        correct_t += 1

    if ground_truth and hypotheses:
        hyp_scores, score_mean = np.array(hyp_scores), np.mean(all_scores)
        ind = np.asarray(hyp_scores >= score_mean).nonzero()[0]

        if args.max_examples > 0:
            survey_fname = 'survey_{}_k{}.json'.format(args.lang, args.max_examples)
            hyp_fname = 'hypothesis_{}_k{}.txt'.format(args.lang, args.max_examples)
            if len(ind) > args.max_examples:
                sample = np.random.choice(ind, size=(args.max_examples,), replace=False)#p=p, replace=False)
            else:
                sample = ind
        else:
            survey_fname = 'survey_{}.json'.format(args.lang)
            hyp_fname = 'hypothesis_{}.txt'.format(args.lang)
            sample = ind

        with open(os.path.join(eval_folder, hyp_fname), 'w') as hp_file,\
             open(os.path.join(eval_folder, survey_fname), 'w') as sm_file:
            items = []
            for j in sample:
                hp_file.write(remove_unicode_punctuation(hypotheses[j]) + '\n')
                items.append(survey["items"][j])
                if survey["gold"]:
                    items.append(survey["gold"][j])
            survey["items"] = items
            del survey["gold"]
            print("Written {} items to the survey".format(len(items)))
            json.dump(survey, sm_file)

        N_ref = max([len(x) for x in ground_truth])
        for i in range(N_ref):
            if args.max_examples > 0:
                gt_fname = 'ground_truth_{}_k{}_{}.txt'.format(args.lang, args.max_examples, i)
            else:
                gt_fname = 'ground_truth_{}_{}.txt'.format(args.lang, i)
            with open(os.path.join(eval_folder, gt_fname), 'w') as gt_file:
                for j in sample:
                    if i < len(ground_truth[j]):
                        gt_file.write(ground_truth[j][i] + '\n')
                    else:
                        gt_file.write('\n')

    eval_file.close()

    if args.modeldb_url:
        run.log_metric("correct_questions", correct_q, overwrite=True)
        run.log_metric("correct_qa_pairs", correct_t, overwrite=True),
        run.log_metric("generated_qa_total", total_gen, overwrite=True),
        run.log_metric("questions_in_corpus", total, overwrite=True),
        run.log_artifact("eval", eval_folder, overwrite=True)
