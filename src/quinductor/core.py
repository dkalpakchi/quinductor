import re
import time
import string
import math
from operator import itemgetter
from itertools import product, permutations
from pprint import pprint
from collections import defaultdict

import numpy as np
import stanza

from .common import repr_pos_morph


START_TOKEN = "<START>"
END_TOKEN = "<END>"


class GeneratedQAPair:
    def __init__(self, q, a, tmpl, score):
        self.__q = q
        self.__a = a
        self.__tmpl = tmpl
        self.__score = score

    @property
    def q(self):
        return self.__q

    @property
    def a(self):
        return self.__a

    @property
    def template(self):
        return self.__tmpl

    @property
    def score(self):
        return self.__score

    def __str__(self):
        return "{} -- {} ({})".format(self.__q, self.__a, self.__score)


def invoke_guards(root, guards_root, return_first=False):
    cand, survivors = [(root.children[0], guards_root.children, [], [])], []
    used_templates = []

    while cand:
        new_cand = []
        for node, clauses, guard_chain, backup in cand:
            if type(clauses) == dict:
                for cname, g in clauses.items():
                    satisfying = g.cond(node, False)#g.parent.is_root())
                    if satisfying:
                        if g.children:
                            for sat in satisfying:
                                gc = list(guard_chain)
                                gc.append(g.rule)
                                new_cand.append((node if guard_chain else sat, g.children, gc, list(g.templates)))
                        elif backup not in used_templates:
                            gc = list(guard_chain)
                            survivors.append((node, backup, gc))
                            used_templates.append(backup)
                        
                        if g.templates and g.templates not in used_templates:
                            gc = list(guard_chain)
                            gc.append(g.rule)
                            survivors.append((node, g.templates, gc))
                            used_templates.append(g.templates)

                            if return_first:
                                return survivors

                    elif backup not in used_templates:
                        gc = list(guard_chain)
                        survivors.append((node, backup, gc))
                        used_templates.append(backup)
        cand = new_cand
    return survivors


def split_coordinate_clauses(tree_root):
    conjuncts = tree_root.select_by("deprel", "conj")
    
    clauses = {}
    for conj in conjuncts:
        clause1_root = conj.parent
        clause2_root = conj

        if clause2_root.child_has_prop("deprel", "nsubj"):
            # means we did find a coordinate clause
            clause1_root.remove_child(clause2_root)
            while clause1_root.deprel != 'root':
                # get to the real root
                clause1_root = clause1_root.parent
            clause2_root.prune('cc')
            clause2_root.make_root()
            clauses[str(clause1_root)] = clause1_root
            clauses[str(clause2_root)] = clause2_root
    return list(clauses.values())


def generate_questions(trees, tools, **kwargs):
    generated = []
    res = overgenerate_questions(
        trees, tools['guards'],
        tools['templates'], tools['examples'], **kwargs
    )

    if res:
        dep_proc = 'tokenize,lemma,mwt,pos,depparse' if tools['lang'] in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
        proc = 'tokenize,mwt,pos' if tools['lang'] in ['fi', 'ar'] else 'tokenize,pos'
        stanza_dep_pipe = stanza.Pipeline(lang=tools['lang'], processors=dep_proc)
        stanza_pipe = stanza.Pipeline(lang=tools['lang'], processors=proc)

        idx_sorted_by_scores, qwf, atf, scores = rank(
            res, stanza_pipe, stanza_dep_pipe, tools['qw_stat'], tools['a_stat'], tools['pos_ngrams'],
            rtl=tools['rtl'], join_char=tools['join_symbol'])

        for i in idx_sorted_by_scores:
            if len(res[i]['answer']) == 1 and atf[i] < 1:
                # if the answer is one word and its sequence of pos-morph tags not appeared in the corpus
                continue

            if qwf[i] == 0:
                # if the combination of the question word and the root token of the answer never appeared in the corpus
                continue

            q = "{}?".format(tools['join_symbol'].join(res[i]['question']))
            a = tools['join_symbol'].join(res[i]['answer'])

            if a.strip() and q.strip():
                generated.append(GeneratedQAPair(q, a, res[i]['temp_id'], scores[i]))

    return generated


def overgenerate_questions(trees, guards_root, templates, template_examples, return_first=False, distractors=False):
    res = {}
    for root in trees:
        survivors = invoke_guards(root, guards_root, return_first=return_first)

        generated_pairs = set()
        for node, guard_templates, guard_chain in survivors:
            for temp_id in guard_templates:
                temp = templates[temp_id]
                qtemp, atemp, shared_tags = temp['question'], temp['answer'], temp['shared_tags']
                q_expressions, a_expressions = qtemp(node), atemp(node)

                # Example expression elements
                #
                # Q: 
                # [{None: TemplateInstance(None, ['var'])},
                #  {None: TemplateInstance(None, ['ligger'])},
                #  {4: TemplateInstance(node<4>, ['tyngdpunkten'])},
                #  {2: TemplateInstance(node<2>, ['i biomedicinarutbildningen']), 9: TemplateInstance(node<9>, ['till kunskaper om människan och hennes sjukdomar'])}]
                # 
                # A:
                # [{6: TemplateInstance(node<6>, ['på molekylärbiologi'])}
                #  {7: TemplateInstance(node<7>, ['kopplat'])},
                #  {2: TemplateInstance(node<2>, ['i biomedicinarutbildningen']), 9: TemplateInstance(node<9>, ['till kunskaper om människan och hennes sjukdomar'])}]

                # the lists are becoming too large - try to get away with generators as much as possible!
                for temp_question in product(*[q.keys() if q else [] for q in q_expressions]):
                    tq = [x for x in temp_question if x is not None]
                    if len(set(tq)) == len(tq):

                        if a_expressions:
                            for temp_answer in product(*[a.keys() if a else [] for a in a_expressions]):
                                ta = [x for x in temp_answer if x is not None]
                                if len(set(ta)) == len(ta):
                                    passing = True
                                    for s in shared_tags:
                                        q_last, a_last = s.last_in_chain('q'), s.last_in_chain('a')
                                        q = q_expressions[s.q][temp_question[s.q]]
                                        a = a_expressions[s.a][temp_answer[s.a]]

                                        q_check = [q.chain[-1]] if q_last else q.chain
                                        a_check = [a.chain[-1]] if a_last else a.chain

                                        if not (set(q_check) & set(a_check)):
                                            passing = False
                                            break

                                    if passing:
                                        for q_seq in product(*[q_expressions[i][k].text for i, k in enumerate(temp_question)]):
                                            for a_seq in product(*[a_expressions[i][k].text for i, k in enumerate(temp_answer)]):
                                                # exclude pronouns for now until pronoun resolution is there
                                                # PUNCT_REGEX = f'[{string.punctuation}]'
                                                # qq = re.sub(PUNCT_REGEX, '', q.lower())
                                                # aa = re.sub(PUNCT_REGEX, '', a.lower())
                                                # if set(qq.split()) & PERSONAL_PRONOUNS or set(aa.split()) & PERSONAL_PRONOUNS:
                                                #     continue

                                                # if any template element was evaluated to an empty sequence
                                                if any([not x.strip() for x in a_seq]) or any([not x.strip() for x in q_seq]):
                                                    continue

                                                el = {
                                                    'question': [x.strip() for x in q_seq],
                                                    'answer': [x.strip() for x in a_seq],
                                                    'temp_id': [str(temp_id)],
                                                    'guards': [", ".join(guard_chain)],
                                                    "base_sentences": [str(root.children[0].get_subtree_text())],
                                                }


                                                if el['answer'] in el['question']:
                                                    # Obviously wrong question!
                                                    continue

                                                if distractors:
                                                    # TODO: fix conv_tree_kernel and then enable again
                                                    # dis = generate_distractors(trees, a)
                                                    dis = []
                                                    el['distractors'] = dis
                                                else:
                                                    el['distractors'] = []
                                                
                                                pair = f"{el['question']} => {el['answer']}"
                                                if pair not in res:
                                                    res[pair] = el
                                                else:
                                                    if el['temp_id'][0] not in res[pair]['temp_id']:
                                                        res[pair]['temp_id'].extend(el['temp_id'])
                                                        res[pair]['guards'].extend(el['guards'])
                                                        res[pair]['base_sentences'].extend(el['base_sentences'])
                        else:
                            for q_seq in product(*[q_expressions[i][k].text for i, k in enumerate(temp_question)]):
                                # exclude pronouns for now until pronoun resolution is there
                                # PUNCT_REGEX = f'[{string.punctuation}]'
                                # qq = re.sub(PUNCT_REGEX, '', q.lower())
                                # aa = re.sub(PUNCT_REGEX, '', a.lower())
                                # if set(qq.split()) & PERSONAL_PRONOUNS or set(aa.split()) & PERSONAL_PRONOUNS:
                                #     continue

                                # if any template element was evaluated to an empty sequence
                                if any([not x.strip() for x in q_seq]):
                                    continue

                                el = {
                                    'question': [x.strip() for x in q_seq],
                                    'answer': [],
                                    'temp_id': [str(temp_id)],
                                    'guards': [", ".join(guard_chain)],
                                    "base_sentences": [str(root.children[0].get_subtree_text())],
                                }
                                
                                pair = f"{el['question']} => {el['answer']}"
                                if pair not in res:
                                    res[pair] = el
                                else:
                                    if el['temp_id'][0] not in res[pair]['temp_id']:
                                        res[pair]['temp_id'].extend(el['temp_id'])
                                        res[pair]['guards'].extend(el['guards'])
                                        res[pair]['base_sentences'].extend(el['base_sentences'])
    return list(res.values())


def get_syntactic_score_old(q_words, unigrams_log_prob, bigrams_log_prob, trigrams_log_prob):
    def backoff(w):
        return w if type(w) == str else w.upos

    lambda4 = 0.000001
    lambda3 = 0.01 - lambda4
    lambda2 = 0.1 - lambda3
    lambda1 = 0.9

    q_words.insert(0, START_TOKEN)
    q_words.append(END_TOKEN)

    N = len(q_words)
    syntactic_score = 0
    if N >= 3:
        N_ngrams = 0
        for i, (w1, w2, w3) in enumerate(zip(q_words[:-2], q_words[1:-1], q_words[2:])):
            w1_token, w2_token, w3_token = repr_pos_morph(w1), repr_pos_morph(w2), repr_pos_morph(w3)
            w1_backoff, w2_backoff, w3_backoff = backoff(w1), backoff(w2), backoff(w3)

            w1u_lprob = unigrams_log_prob.get(w1_token, unigrams_log_prob.get(w1_backoff, float('-inf')))
            
            w1b_lprob = bigrams_log_prob.get(w1_token, bigrams_log_prob.get(w1_backoff, {}))
            w1w2b_lprob = w1b_lprob.get(w2_token, w1b_lprob.get(w2_backoff, float('-inf')))

            w1t_lprob = trigrams_log_prob.get(w1_token, trigrams_log_prob.get(w1_backoff, {}))
            w1w2t_lprob = w1t_lprob.get(w2_token, w1t_lprob.get(w2_backoff, {}))
            w1w2w3t_lprob = w1w2t_lprob.get(w3_token, w1w2t_lprob.get(w3_backoff, float('-inf')))
            N_ngrams += 1

            syntactic_score += np.log(lambda1 * np.exp(w1w2w3t_lprob) + lambda2 * np.exp(w1w2b_lprob) + lambda3 * np.exp(w1u_lprob) + lambda4)
        syntactic_score /= N_ngrams
    elif N == 2:
        w1, w2 = q_words[0], q_words[1]
        w1_token, w2_token = repr_pos_morph(w1), repr_pos_morph(w2)
        w1_backoff, w2_backoff = backoff(w1), backoff(w2)
        w1u_lprob = unigrams_log_prob.get(w1_token, unigrams_log_prob.get(w1_backoff))
        w1b_lprob = bigrams_log_prob.get(w1_token, bigrams_log_prob.get(w1_backoff))
        w1w2b_lprob = w1b_lprob.get(w2_token, w1b_lprob.get(w2_backoff, float('-inf')))
        syntactic_score += np.log(lambda1 * np.exp(w1w2b_lprob) + lambda2 * np.exp(w1u_lprob) + (1 - lambda1 - lambda2))
        syntactic_score /= 2
    elif N == 1:
        w1 = q_words[0]
        w1_token = repr_pos_morph(w1)
        syntactic_score += np.log(lambda1 * np.exp(unigrams_log_prob.get(w1_token, unigrams_log_prob.get(w1.upos, float('-inf')))) + (1 - lambda1))
    return np.exp(syntactic_score)


def get_syntactic_score(q_words, unigrams_log_prob, bigrams_log_prob, trigrams_log_prob):
    def backoff(w):
        return w if type(w) == str else w.upos

    lambda4 = 0.000001
    lambda3 = 0.01 - lambda4
    lambda2 = 0.1 - lambda3
    lambda1 = 0.9

    q_words.insert(0, START_TOKEN)
    q_words.append(END_TOKEN)

    N = len(q_words)
    syntactic_score = 0
    if N >= 3:
        N_ngrams = 0
        for i, w in enumerate(q_words):
            w3_token, w3_backoff = repr_pos_morph(w), backoff(w)
            if i - 1 >= 0:
                w2_token = repr_pos_morph(q_words[i-1])
                w2_backoff = backoff(q_words[i-1])
            else:
                w2_token, w2_backoff = None, None
            
            if i - 2 >= 0:
                w1_token = repr_pos_morph(q_words[i-2])
                w1_backoff = backoff(q_words[i-2])
            else:
                w1_token, w1_backoff = None, None

            w3u_lprob = unigrams_log_prob.get(w3_token, unigrams_log_prob.get(w3_backoff, float('-inf')))
            
            if w2_token:
                w2b_lprob = bigrams_log_prob.get(w2_token, bigrams_log_prob.get(w2_backoff, {}))
                w2w3b_lprob = w2b_lprob.get(w3_token, w2b_lprob.get(w3_backoff, float('-inf')))
            else:
                w2w3b_lprob = float('-inf')

            if w1_token:
                w1t_lprob = trigrams_log_prob.get(w1_token, trigrams_log_prob.get(w1_backoff, {}))
                w1w2t_lprob = w1t_lprob.get(w2_token, w1t_lprob.get(w2_backoff, {}))
                w1w2w3t_lprob = w1w2t_lprob.get(w3_token, w1w2t_lprob.get(w3_backoff, float('-inf')))
            else:
                w1w2w3t_lprob = float('-inf')
            
            N_ngrams += 1

            # some versions of Numpy give `FloatingPointError: underflow encountered in exp`
            # when the argument is float('-inf'), hence this workaround
            w1w2w3t_prob = 0 if w1w2w3t_lprob == float('-inf') else np.exp(w1w2w3t_lprob)
            w2w3b_prob = 0 if w2w3b_lprob == float('-inf') else np.exp(w2w3b_lprob)
            w3u_prob = 0 if w3u_lprob == float('-inf') else np.exp(w3u_lprob)

            syntactic_score += np.log(lambda1 * w1w2w3t_prob + lambda2 * w2w3b_prob + lambda3 * w3u_prob + lambda4)
        syntactic_score /= N_ngrams
    elif N == 2:
        w1, w2 = q_words[0], q_words[1]
        w1_token, w2_token = repr_pos_morph(w1), repr_pos_morph(w2)
        w1_backoff, w2_backoff = backoff(w1), backoff(w2)
        w2u_lprob = unigrams_log_prob.get(w2_token, unigrams_log_prob.get(w2_backoff))
        w1b_lprob = bigrams_log_prob.get(w1_token, bigrams_log_prob.get(w1_backoff))
        w1w2b_lprob = w1b_lprob.get(w2_token, w1b_lprob.get(w2_backoff, float('-inf')))

        w1w2b_prob = 0 if w1w2b_lprob == float('-inf') else np.exp(w1w2b_lprob)
        w2u_prob = 0 if w2u_lprob == float('-inf') else np.exp(w2u_lprob)

        syntactic_score += np.log(lambda1 * w1w2b_prob + lambda2 * w2u_prob + (1 - lambda1 - lambda2))
        syntactic_score /= 2
    elif N == 1:
        w1 = q_words[0]
        w1_token = repr_pos_morph(w1)
        w1u_lprob = unigrams_log_prob.get(w1_token, unigrams_log_prob.get(w1.upos, float('-inf')))

        w1u_prob = 0 if w1u_lprob == float('-inf') else np.exp(w1u_lprob)

        syntactic_score += np.log(lambda1 * w1u_prob + (1 - lambda1))
    return np.exp(syntactic_score)


def rank(res, stanza_pipe, stanza_dep_pipe, qw_stat, a_tmpl, log_prob, rtl=False, join_char=' '):
    """
    Rank the overgenerated questions
    
    :param      res:              The list of overgenerated questions
    :type       res:              list
    :param      stanza_pipe:      Stanza pipeline with processors "tokenize,pos"
    :type       stanza_pipe:      stanza.Pipeline
    :param      stanza_dep_pipe:  Stanza pipeline with processors "tokenize,pos,lemma,depparse"
    :type       stanza_dep_pipe:  stanza.Pipeline
    """
    unigrams_log_prob, bigrams_log_prob, trigrams_log_prob = log_prob
    generated_templates = set()
    atemp_freq, qw_freq = [], []
    q_synt, q_sem, qw_freq, atemp_freq = [], [], [], []
    for cand in res:
        # each cand contains question and answer divided into tokens as in the template
        # but those might not necessarily be correct for languages like Japanese or Chinese
        q = join_char.join(cand['question']).strip()
        a = join_char.join(cand['answer']).strip()
        if not q:
            # to preserve indices in `synt` as in `res`
            q_synt.append(float('-inf'))
            qw_freq.append(0)
            atemp_freq.append(0)
            continue

        q_words = stanza_pipe(q).sentences[0].words
        qw_key = (q_words[-1] if rtl else q_words[0]).text.lower()

        if a:
            a_words = stanza_dep_pipe(a).sentences[0].words
            aw = None
            for i, w in enumerate(a_words):
                if w.deprel == 'root':
                    aw = i
                    break
            if aw is None:
                q_synt.append(float('-inf'))
                qw_freq.append(0)
                atemp_freq.append(0)
                continue

            a_pos_pattern = [repr_pos_morph(x) for x in a_words]
        
            atemp_freq.append(a_tmpl[" ".join(a_pos_pattern)])

            if '/' in a_pos_pattern[aw]:
                a_pos, a_morph = a_pos_pattern[aw].split('/')
                a_morph = set(a_morph.split('|'))
            else:
                a_pos, a_morph = a_pos_pattern[aw], None

            valid_freq, total_freq = 0, 0
            for pos_pattern, freq in qw_stat[qw_key].items():
                if '/' in pos_pattern:
                    pos, morph = pos_pattern.split('/')
                    morph = morph.split('|')
                    if a_pos == pos and a_morph and a_morph & set(morph) == a_morph:
                        valid_freq += freq
                else:
                    if a_pos == pos_pattern:
                        valid_freq += freq
                total_freq += freq

            if total_freq > 0:
                qw_freq.append(round(valid_freq / total_freq, 4))
            else:
                qw_freq.append(0)
        else:
            # these are to ignore the atf constraint, since we just have no answer
            qw_freq.append(1)
            atemp_freq.append(1)

        if qw_key not in qw_stat:
            if rtl:
                cand['question'] = " ".join([x.text for x in q_words[:-1]]) + qw_key
            else:
                cand['question'] = qw_key + " " + " ".join([x.text for x in q_words[1:]])

        q_synt.append(round(get_syntactic_score(q_words, unigrams_log_prob, bigrams_log_prob, trigrams_log_prob), 4))

    scores = [0.8 * q_synt[x] + 0.2 * qw_freq[x] for x in range(len(res))]

    idx_sorted_by_scores = sorted(
        list(range(len(res))),
        key=lambda x: scores[x],
        reverse=True
    )
    return idx_sorted_by_scores, qw_freq, atemp_freq, scores