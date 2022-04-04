# -*- coding: utf-8 -*-
import json
import dill
import argparse
import os
import math
import sys
import time
import string
import shutil
import re
import itertools
from collections import defaultdict
from operator import itemgetter
from difflib import SequenceMatcher

import tqdm

import stanza
from stanza.utils.conll import CoNLL

import udon2

from .loaders import *
from .repro import get_client
from .common import *
from .guards import *
from .rules import load_templates


logger = get_logger()


def get_negatives(node, chain):
    # get all possible negatives, which is all children of the node and the node itself if it has any children
    negatives = [chain + [f"{c.deprel}#{int(c.id)}"] for c in node.children]
    if chain and len(node.children) > 0:
        node_el = list(chain)
        node_el[-1] += "*"
        negatives.append(node_el)
    return negatives


def shift_reduce(template_elements):
    template, children_map = [], {}
    N = len(template_elements)
    for t in template_elements:
        logger.debug(" ".join(map(str, template)) + " <-- " + str(t))
        if template:
            logger.debug("type(template[-1]): {}".format(type(template[-1])))
            if type(template[-1]) == TemplateElement:
                logger.debug("\ttemplate[-1] is lemma? -- {}".format(template[-1].is_lemma))
        logger.debug("type(t): {}".format(type(t)))
        if type(t) == TemplateElement:
            logger.debug("\tt is lemma? -- {}".format(t.is_lemma))
        if template and type(template[-1]) == TemplateElement and type(t) == TemplateElement and not template[-1].is_lemma and not t.is_lemma:
            logger.debug("stack contains at least two items!")
            intersection = template[-1] & t
            N1, N2, Ni = len(template[-1]), len(t), len(intersection)
            logger.debug("intersection: {}".format(str(intersection)))
            logger.debug("N1: {}, N2: {}, Ni: {}".format(N1, N2, Ni))

            if intersection and (N1 - Ni <= 1 or N2 - Ni <= 1):
                # If the intersection exists and it's one level above any of the two candidates,
                # we can collapse them then. The restriction on the levels is needed
                # to avoid huge collapses that will probably be non-generalizable, i.e.
                # <w.conj - obl5*>, whereas the conj structures can vary a lot and hence
                # some wrong questions will be generated
                
                logger.debug("==> REDUCE")

                negatives = []
                for node in [t, template[-1]]:
                    new_node, new_chain = node.node, node.chain
                    new_chain_str = ".".join(new_chain)
                    if new_chain_str not in children_map:
                        new_neg = get_negatives(new_node, new_chain)
                        children_map[new_chain_str] = new_neg
                    negatives.extend(children_map[new_chain_str])

                    while new_chain != intersection:
                        new_node = new_node.parent
                        new_chain = new_chain[:-1]

                        new_chain_str = ".".join(new_chain)
                        if new_chain_str not in children_map:
                            new_neg = get_negatives(new_node, new_chain)
                            children_map[new_chain_str] = new_neg
                        negatives.extend(children_map[new_chain_str])

                logger.debug("\tnegatives: {}".format(negatives))

                # means they are part of the same subtree
                te = TemplateElement(intersection, new_node, t.id, True)
                for n in [template[-1], t]:
                    if len(n.node.children) > 0 and not n.is_subtree:
                        n.chain[-1] += "*"

                te.copy_positives([template[-1], t]) # copy only positives?
                te.add_positives([template[-1].chain, t.chain]) # if those are copied, then this addition is not necessary?
                te.copy_negatives([template[-1], t])
                te.add_negatives(negatives)

                logger.debug("\treduced token: {}".format(str(te)))

                template.pop()
                template.append(te)
            else:
                logger.debug("==> SHIFT: the token {} is added to stack".format(str(t)))
                # means they are from different subtrees
                template.append(t)
        else:
            logger.debug("==> SHIFT: the token {} is added to stack".format(str(t)))
            # just the second word
            template.append(t)
    return template, children_map


def generate_question_template(s_root, q_root, strict=True, join_char=' '):
    """
    Generates question template by first linearizing the question dep parse tree
    and then trying to get a question word from the beginning nodes of the linear order.

    If the node `n` is the question root itself, add a node template element [w]
     
    :param      s_root:  The node representing the root word of the original sentence
    :type       s_root:  udon2.Node
    :param      q_root:  The node representing the root word of the question
    :type       q_root:  udon2.Node
    """
    node_list = q_root.linear()
    N = len(node_list)

    logger.debug("-- S-TRANSFORM --")
    template_elements, is_template_element = [], []
    for i in range(N):
        # can't use `node_list[i].get_rel() != "punct"`, since sometimes adpositions become punct!
        if node_list[i].form not in string.punctuation:
            if s_root.form.lower() == node_list[i].form.lower():
                # means it's a root of the sentence tree itself
                template_elements.append([TemplateElement([], s_root)])
                is_template_element.append(True)
            elif s_root.lemma == node_list[i].form.lower():
                template_elements.append([TemplateElement([], s_root, is_lemma=True)])
                is_template_element.append(True)
            else:
                # dependency relation can be different and POS-tag as well (e.g. "har" can be both AUX and VERB)
                # morphological properties should typically be the same though
                s_subtree = s_root.select_by("form", node_list[i].form) # by lemma and text
                if len(s_subtree) == 0:
                    s_subtree = s_root.select_by("form", node_list[i].form.lower())
                
                if len(s_subtree) == 0:
                    # try to match by lemma for the cases of "When does he play football?" and the sentence "He plays football every week."
                    s_lemma = s_root.select_by("lemma", node_list[i].form.lower())

                    if len(s_lemma) == 0:
                        if not strict:
                            template_elements.append([node_list[i].form])
                            is_template_element.append(False)
                        else:
                            # print("0-length!")
                            # means we can't find one of the words!
                            return None, False, False
                    else:
                        candidates = []
                        for cand in s_lemma:
                            root_chain = chain(s_root, cand, include_ids=True)
                            candidates.append(TemplateElement(root_chain, cand, is_lemma=True))

                        template_elements.append(candidates)
                        is_template_element.append(True)
                elif len(s_subtree) == 1:
                    cand = s_subtree[0]
                    root_chain = chain(s_root, cand, include_ids=True)
                    template_elements.append([TemplateElement(root_chain, cand)])
                    is_template_element.append(True)
                else:
                    candidates = []
                    for cand in s_subtree:
                        root_chain = chain(s_root, cand, include_ids=True)
                        candidates.append(TemplateElement(root_chain, cand))

                    if not candidates:
                        # print("0-cand")
                        # we don't want to work with those not connected to question phrase
                        # where we can't find the candidate connected to the parent
                        return None, False, False

                    template_elements.append(candidates)
                    is_template_element.append(True)

    min_diff, best_cand = float('inf'), None

    pair_check = list(zip(is_template_element[:-1], is_template_element[1:]))

    cand_processed, cand_limit = 0, 200 # computational simplification for processing potentially long sentences
    # problem with the number of hops: how would you calculate the number of hops if they are on the different side of the root?
    # sum before the root + sum after the root? not exactly clear
    for cand in itertools.product(*template_elements):
        if cand_processed >= cand_limit: break
        if len(set(cand)) == len(cand):
            diff = sum([abs(cand[i+1].id - cand[i].id) for i, (n1, n2) in enumerate(pair_check)
                if n1 and n2 and cand[i].id is not None and cand[i+1].id is not None])
            if diff < min_diff:
                min_diff = diff
                best_cand = cand
        cand_processed += 1

    if best_cand is None and min_diff == float('inf'):
        best_cand = cand

    logger.debug(list(map(str, best_cand)))
    
    logger.debug("-- SHIFT-REDUCE --")

    template, children_map = shift_reduce(best_cand)

    logger.debug(" ".join(map(str, template)))

    if len(template) == 1:
        logger.debug("-- OVERGENERIC! RETURN S-TRANSFORM --")
        # overgeneric case, like what <w.obl - case - num>?
        # simply go on with an original word by word template after S-transform
        best_cand = list(best_cand)
        S_t = sum([type(x) == TemplateElement for x in best_cand])
        return best_cand

    logger.debug("-- MERGING NEGATIVES --")
    for t in template:
        if type(t) == TemplateElement:
            t.merge_negatives(children_map)

    logger.debug(join_char.join(map(str, template)))
    logger.debug("-- END OF TRANSFORMATION --\n")
    return template


def generate_answer_template(s_root_word, answer_str, join_char=' '):
    # try to find the right subtree containing the answer_str
    answer_node = s_root_word.textual_intersect(answer_str.strip())
    logger.debug(f"-- ANSWER STRING -> {answer_str} --")
    
    if answer_node:
        logger.debug(f"-- FOUND ANSWER NODE -> {answer_node.form} --")
        logger.debug(f"-- SUBTREE ANSWER NODE -> {answer_node.get_subtree_text()} --")
    else:
        # this case should not happen normally for Swedish, because we know for sure
        # the answer is in the original sentence - that's how the dataset is structured
        # but for other datasets it's definitely a possibility
        logger.debug("-- ANSWER NOT PRESENT --\n")
        return None

    # find the way from the answer node of the original sentence to its root
    if answer_node.form.lower() == answer_str.lower():
        root_chain = chain(s_root_word, answer_node, include_ids=True)
        template = [TemplateElement(root_chain, answer_node)]
        logger.debug(" ".join(map(str, template)))
        logger.debug("-- END OF TRANSFORMATION --\n")
    elif answer_node.get_subtree_text().lower() == answer_str.lower():
        root_chain = chain(s_root_word, answer_node, include_ids=True)
        template = [TemplateElement(root_chain, answer_node, is_subtree=True)]
        logger.debug(" ".join(map(str, template)))
        logger.debug("-- END OF TRANSFORMATION --\n")
    else:
        sketch, start = [], None
        for i, word in enumerate(re.findall(r"[\w']+|[.,!?;]", answer_str.strip())):
            res = answer_node.select_by("form", word)
            if len(res) == 1:
                if not start: start = i
                root_chain = chain(s_root_word, res[0], include_ids=True)
                sketch.append(TemplateElement(root_chain, res[0]))
            elif res:
                res2 = []
                for r in res:
                    root_chain = chain(s_root_word, r, include_ids=True)
                    res2.append(TemplateElement(root_chain, r))
                sketch.append(res2)
            else:
                logger.debug("-- COULDN'T FIND: res, word --")
                logger.debug("-- NO ANSWER COULD BE GENERATED --\n")
                return None

        logger.debug(f"-- START -> {start} --")

        if not start:
            logger.debug("-- NO ANSWER COULD BE GENERATED --\n")
            return None

        # we know for sure that at least one word will match exactly
        for j in list(range(0, start)) + list(range(start + 1, i + 1)):
            if type(sketch[j]) == list:
                min_dist, best_match = float('inf'), None
                for el in sketch[j]:
                    if j > 0 and el.id - sketch[j-1].id == 1:
                        # case of the exact following
                        best_match = el
                        break

                    d = abs(el.id - sketch[start].id)
                    if d < min_dist:
                        min_dist = d
                        best_match = el
                sketch[j] = best_match
        
        logger.debug(f"ANSWER SKETCH --- {list(map(str, sketch))}")

        template, children_map = shift_reduce(sketch)

        for t in template:
            if type(t) == TemplateElement:
                t.merge_negatives(children_map)
        
        logger.debug(join_char.join(map(str, template)))
        logger.debug("-- END OF TRANSFORMATION --\n")

    return template


def process_modifiers(templates):
    """
    If there are any modifiers (advmod or amod), generate identical templates, but without
    these modifiers
    
    :param      templates:  The templates
    :type       templates:  dict
    """

    to_process = list(templates.items())
    search_temp = [f"<w.{x}>" for x in MODIFIER_RELS]
    for temp, sent in to_process:
        tokens = temp.split()
        new_tokens = []
        answer_started = False
        for token in tokens:
            if token == "=>":
                answer_started = True
                new_tokens.append(token)
            elif token not in search_temp or answer_started:
                new_tokens.append(token)
        templates[" ".join(new_tokens)] = sent
    return templates


def replace_root_by_chain(temp, chain_str):
    return temp.replace("w.", f"w.{chain_str}.")\
               .replace('<w>', f"<w.{chain_str}>")\
               .replace('[w]', f"[w.{chain_str}]")


def normalize_templates(qtemp, atemp):
    def __norm_temp(x):
        nonlocal idx, mapping
        for i in range(len(x)):
            if type(x[i]) == TemplateElement and x[i].chain:
                new_chain = []
                for el in x[i].chain:
                    if el not in mapping:
                        mapping[el] = f"{el.split('#')[0]}#{idx}"
                        idx += 1
                    new_chain.append(mapping[el])

                positives = []
                kept = set()
                for lst in x[i].positives:
                    new_lst = []
                    for j in range(len(lst)):
                        exact = '*' in lst[j]
                        lst_j = lst[j].replace('*', '')
                        if lst_j in mapping:
                            new_lst.append(mapping[lst_j] + ('*' if exact else ''))
                        else:
                            kept.add(lst[j])
                            new_lst.append(lst[j])
                    positives.append(new_lst)

                negatives = []
                for lst in x[i].negatives:
                    new_lst = []
                    for j in range(len(lst)):
                        if lst[j] in kept:
                            new_lst.append(lst[j])
                        else:
                            exact = '*' in lst[j]
                            lst_j = lst[j].replace('*', '')
                            if lst_j not in mapping:
                                mapping[lst_j] = f"{lst_j.split('#')[0]}#{idx}"
                                idx += 1

                            new_lst.append(mapping[lst_j] + ('*' if exact else ''))
                    negatives.append(new_lst)

                new_el = TemplateElement(new_chain, x[i].node, last_span_id=x[i].id, is_subtree=x[i].is_subtree, is_lemma=x[i].is_lemma)
                new_el.add_positives(positives)
                new_el.add_negatives(negatives)

                x[i] = new_el
        return x

    idx = 1
    mapping = {}
    return __norm_temp(qtemp), __norm_temp(atemp)


def generate_templates(fname, stanza_lang, rtl=False, min_support=2, strict=True, case_folding=False, remove_punct=False,
                       temp_fname="gen_templates.txt", sent_fname="gen_sentences.txt", remove_diacritics=True, dot_fix=False,
                       join_char=' ', idf_file=None, no_answers=False):
    def record(key):
        problems[key] += 1

    def print_report(templates):
        print("{} templates".format(len(templates)))
        print("{} impossible questions".format(problems['impossible']))
        print("{} possible questions of which {} share the root with the original sentence".format(problems['possible'], problems['same_root']))
        print("{} copula questions".format(problems['copula']))
        print("{} have no question templates".format(problems['no_q_template']))
        print("{} have no answer templates".format(problems['no_a_template']))

    problems = {
        'possible': 0,
        'impossible': 0,
        'same_root': 0,
        'copula': 0,
        'no_q_template': 0,
        'no_a_template': 0
    }

    templates = {}
    with open(fname) as f:
        for line in tqdm.tqdm(f):
            if line.strip():
                question, answer, base_sentence = line.split(" #|@ ")

                if remove_diacritics:
                    question = remove_unicode_diacritics(question)
                    base_sentence = remove_unicode_diacritics(base_sentence)
                    answer = remove_unicode_diacritics(answer)

                if case_folding:
                    question, answer, base_sentence = question.lower(), answer.lower(), base_sentence.lower()

                # lowercasing is a necessary step to mitigate parser's errors
                if remove_punct:
                    question = remove_unicode_punctuation(question)
                    base_sentence = remove_unicode_punctuation(base_sentence)
                    answer = remove_unicode_punctuation(answer)

                question, base_sentence, answer = question.strip(), base_sentence.strip(), answer.strip()
                if dot_fix:
                    if not is_punctuation(question[-1]):
                        question += "?"
                    if not is_punctuation(base_sentence[-1]):
                        base_sentence += "."

                # have to proceed through files, because C++ package works with files
                with open('sentence.conll', 'w') as f1:
                    ss = stanza_lang(base_sentence)
                    conll_list = CoNLL.convert_dict(ss.to_dict())
                    sentence_tokenized = [w.text for s in ss.sentences for w in s.words]
                    f1.write(CoNLL.conll_as_string(conll_list))

                with open('question.conll', 'w') as f1:
                    qq = stanza_lang(question)
                    conll_list = CoNLL.convert_dict(qq.to_dict())
                    question_tokenized = [w.text for s in qq.sentences for w in s.words]
                    f1.write(CoNLL.conll_as_string(conll_list))

                ud_s = udon2.ConllReader.read_file('sentence.conll')[0]
                ud_q = udon2.ConllReader.read_file('question.conll')[0]
                
                # s_roots = udon2.ConllReader.read_file('sentence.conll')
                # q_roots = udon2.ConllReader.read_file('question.conll')
                # ud_s = s_roots[0]
                # ud_q = q_roots[0]

                s_root_word = ud_s.children[0]
                q_root_word = ud_q.children[0]

                if strict:
                    diff = get_difference(question_tokenized, sentence_tokenized)
                    cond = not diff
                else:
                    same = get_intersection(question_tokenized, sentence_tokenized)
                    cond = len(same) > 0
                
                if cond:
                    # means there's a direct dependency tree transformation!
                    record('possible')
                    if s_root_word.form.lower() == q_root_word.form.lower():
                        # many questions that can be asked share the root with a sentence
                        record('same_root')
                    elif q_root_word.prop_exists("deprel", "cop"):
                        # means this is a copula question
                        record('copula')

                    q_temp = generate_question_template(s_root_word, q_root_word, strict=strict, join_char=join_char)

                    to_check = q_temp[:-1] if rtl else q_temp[1:]
                    S_t = sum([type(x) == TemplateElement for x in to_check])
                    S_nt = len(to_check) - S_t

                    if not q_temp:
                        record('no_q_template')
                        continue
                    if S_t == 0:
                        continue
                    
                    qw = q_temp[-1] if rtl else q_temp[0]
                    if type(qw) == TemplateElement:
                        # the first word is not a constant, so no question word there
                        continue
                    
                    if rtl:
                        qw = q_temp.pop()
                        q_temp.append('<qw>')
                    else:
                        qw = q_temp.pop(0)
                        q_temp.insert(0, '<qw>')

                    if answer and not no_answers:
                        a_temp = generate_answer_template(s_root_word, answer, join_char=join_char)
                        if not a_temp:
                            record('no_a_template')
                            continue

                        q_temp, a_temp = normalize_templates(q_temp, a_temp)
                    else:
                        a_temp = ''
                    
                    qtemp_without_qw = join_char.join(map(str, q_temp))

                    if qtemp_without_qw not in templates:
                        templates[qtemp_without_qw] = {
                            'question': q_temp,
                            'all_templates': S_nt == 0, 
                            'answer': a_temp,
                            'qw': {}
                        }

                    assert templates[qtemp_without_qw]['all_templates'] == (S_nt == 0), "Inconsistency in templates found"

                    if qw not in templates[qtemp_without_qw]['qw']:
                        templates[qtemp_without_qw]['qw'][qw] = {}

                    atemp_str = join_char.join(map(str, a_temp))
                    if atemp_str not in templates[qtemp_without_qw]['qw'][qw]:
                        templates[qtemp_without_qw]['qw'][qw][atemp_str] = {
                            'answer': a_temp,
                            'examples': []
                        }

                    templates[qtemp_without_qw]['qw'][qw][atemp_str]['examples'].append({
                        'sentence': base_sentence.strip(),
                        'question': question.strip(),
                        'answer': answer.strip(),
                        'node': s_root_word.copy(), # If not copying, then we'll have a memory error, since the associated TreeList will be freed
                    })

                    # templates[f"{s_root_word.upos} #|@ {str(s_root_word.feats)} #|@ {s_root_word.child_has_prop('deprel', 'aux')} #|@ {non_temp_el} #|@ {q_temp}"][qw][a_temp].add(
                    #     base_sentence.strip() + " | " + question.strip() + " | " + answer.strip())
                elif strict:
                    record('impossible')

    idf = load_idf(idf_file) if idf_file else None

    final_templates, temp_id = [], 1
    temp_base = os.path.splitext(os.path.basename(temp_fname))[0]
    with open(temp_fname, "w") as f, open(sent_fname, 'w') as f1:
        for _, passport in templates.items():
            N_ex = sum([len(data['examples']) for _, endings in passport['qw'].items() for _, data in endings.items()])

            q_tmpl = join_char.join(map(str, passport['question']))

            if passport['all_templates'] or N_ex >= min_support:
                idfs = [idf.get(t, float('inf')) for t in passport['question'] if type(t) == str and t != '<qw>']
                max_idf = max(idfs) if idfs else 0
                if max_idf <= math.log(4): # appeared in at least 25% of the documents
                    for qw, endings in passport['qw'].items():
                        for a_tmpl, data in endings.items():
                            logger.debug("-- {} - {} - {} -> PASSED --".format(q_tmpl, passport['all_templates'], N_ex))

                            final_templates.append({
                                'question': q_tmpl.replace("<qw>", qw),
                                'answer': a_tmpl,
                                'props': [
                                    {
                                        'pos': x['node'].upos,
                                        'has_aux': x['node'].child_has_prop('deprel', 'aux'),
                                        'feats': x['node'].feats
                                    } for x in data['examples']
                                ]
                            })

                            sent = "\n".join([" | ".join([x['sentence'], x['question'], x['answer']]) for x in data['examples']])

                            tmpl = "{} => {}".format(q_tmpl.replace('<qw>', qw), a_tmpl)
                            f.write("{}\n".format(tmpl))
                            f1.write("id: {}{}\n{}\n\n".format(temp_base, temp_id, sent))
                            temp_id += 1
                else:
                    logger.debug("-- {} - {} - {} -> FAILED IDF ({}) --".format(q_tmpl, passport['all_templates'], N_ex, max_idf))
            else:
                logger.debug("-- {} - {} - {} -> FAILED --".format(q_tmpl, passport['all_templates'], N_ex))

    print_report(final_templates)

    return final_templates, temp_fname


def write_answer_transformations(transformations, fname):
    with open(fname, 'w') as f:
        for i, (ca, comment) in enumerate(transformations.items()):
            f.write(f"#CASE{i+1}\n")
            f.write(f"CA\t{ca}\n")
            f.write(f"COM\t{comment}\n")
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='A language for template generation (en, sv are currently supported)')
    parser.add_argument('-d', '--data', type=str, help='Comma-separated list of files to generate questions from')
    parser.add_argument('-ft', '--format', type=str, help='Data format (tt for Textinator or squad for Squad)')
    parser.add_argument('-k', '--min-support', default=1, type=int, help="""
        A minimum number of sentences (excluding the question word) required for the template to be included.
        Applied only to the templates with at least one constant expression (excluding the first element, which
        is assumed to be the question word).""")
    parser.add_argument('-st', '--strict', action='store_true') 
    parser.add_argument('-f', '--force', action='store_true', help='Re-read data and ignore cache')
    parser.add_argument('-t', '--templates', type=str, default='NA', help='A template file to generate guards for')
    parser.add_argument('-cf', '--case-folding', action='store_true', help='Lowercase everything if true')
    parser.add_argument('-rp', '--remove-punct', action='store_true', help='Remove punctuation if true')
    parser.add_argument('-mdb', '--modeldb-url', type=str, default='')
    parser.add_argument('-tdir', '--template-dir', type=str, default='tmpl', help='A directory for storing templates')
    parser.add_argument('-rtl', '--right-to-left', action='store_true')
    parser.add_argument('-rd', '--remove-diacritics', action='store_true')
    parser.add_argument('-idf', '--idf', type=str, default='')
    parser.add_argument('--dot-fix', action='store_true', help="""
        Append dots at the end of the sentences even if punctuation is removed in attempt to fix dependency trees""")
    parser.add_argument('-j', '--join-char', type=str, default=" ")
    parser.add_argument('-qac', '--qac-file', type=str, help='A preprocessed .qac file for the given dataset')
    parser.add_argument('-na', '--no-answers', action='store_true',
        help="Whether to generate templates for answers")
    args = parser.parse_args()

    if not args.idf:
        default_file = os.path.join(DEFAULT_TEMPLATES_DIR, args.lang, 'idf_{}.csv'.format(args.lang))
        if os.path.exists(default_file):
            args.idf = default_file
        else:
            logger.error(
                """No valid IDF file was provided. Either download it by running `quinductor.download({})`
                or create your own and supply it using `-idf` argument""".format(args.lang)
            )
            sys.exit(1)

    # arabic, finnish - include mwt
    # russian - exclude mwt
    dep_proc = 'tokenize,lemma,mwt,pos,depparse' if args.lang in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
    proc = 'tokenize,mwt,pos' if args.lang in ['fi', 'ar'] else 'tokenize,pos'

    start = time.time()
    timestamp = str(start).replace('.', '')

    if args.modeldb_url:
        client = get_client(args.modeldb_url)
        client.set_project("SQuAD")#"TyDiQA QG")#"SvQG")
        client.set_experiment(args.lang)
        run = client.set_experiment_run("{}_{}".format(args.lang, timestamp))
        join_char = args.join_char
        if join_char == ' ':
            join_char = '<space>'
        elif join_char == '':
            join_char = '<nothing>'

        run.log_hyperparameters({
            'min_support': args.min_support,
            'data': args.data,
            'strict': args.strict,
            'case_folding': args.case_folding,
            'remove_punct': args.remove_punct,
            'right_to_left': args.right_to_left,
            'stanza_processors': dep_proc,
            'dot_fix': args.dot_fix,
            'min_support': args.min_support,
            'remove_diacritics': args.remove_diacritics,
            'join_character': join_char,
            'idf': args.idf
        })

    questions_fnames = [x.strip() for x in args.data.split(',')]

    DIR = os.path.join(args.template_dir, args.lang, timestamp)
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    gen_fname = os.path.join(DIR, f'{args.lang}.qac')

    if not os.path.exists(gen_fname) or args.force:
        if args.qac_file:
            shutil.copy(args.qac_file, gen_fname)
        else:
            stanza_tokenizer = stanza.Pipeline(lang=args.lang, processors=proc)

            if args.format == 'tt':
                data_loader = TextinatorLoader
            elif args.format == 'squad':
                data_loader = SquadLoader
            elif args.format == 'tydiqa':
                data_loader = TyDiQaLoader
            else:
                # generic case
                data_loader = JsonLinesLoader

            if data_loader == JsonLinesLoader:
                with open(gen_fname, 'w') as f:
                    for q, a, s in data_loader.from_files(questions_fnames, args.lang):
                        f.write(f'{q} #|@ {a} #|@ {s}\n')
            else:
                with open(gen_fname, 'w') as f:
                    for q, a, c in data_loader.from_files(questions_fnames, args.lang):
                        sent = get_sentence_by_answer(a, c, stanza_tokenizer)
                        if sent:
                            f.write(f'{q} #|@ {a["text"]} #|@ {sent}\n')

    if args.templates == 'NA':
        stanza_lang = stanza.Pipeline(lang=args.lang, processors=dep_proc)
        templates, temp_fname = generate_templates(gen_fname, stanza_lang, min_support=args.min_support, strict=args.strict,
            rtl=args.right_to_left, case_folding=args.case_folding, remove_punct=args.remove_punct,
            temp_fname=os.path.join(DIR, "templates.txt"), sent_fname=os.path.join(DIR, "sentences.txt"),
            remove_diacritics=args.remove_diacritics, dot_fix=args.dot_fix, join_char=args.join_char,
            idf_file=args.idf, no_answers=args.no_answers)
    else:
        print("Loading templates from {}".format(args.templates))
        templates, temp_fname = templates = load_templates([args.templates]), args.templates

    guards = generate_guards(templates, temp_fname, guard_fname=os.path.join(DIR, "guards.txt")) if templates else []
    generation_time = round(time.time() - start, 2)
    print(f"Generation took {generation_time} seconds")

    if args.modeldb_url:
        run.log_observation("number_of_templates", len(templates))
        run.log_observation("number_of_guards", len(guards))
        run.log_observation("generation_time_in_seconds", generation_time)
        run.log_artifact("model", DIR)
    