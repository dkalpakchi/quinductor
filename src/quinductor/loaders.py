import os
import json
import operator
import jsonlines
from collections import defaultdict

from pprint import pprint


class TextinatorLoader:
    @staticmethod
    def get_correct_answer(choices):
        return list(filter(lambda x: x['type'] == 'Correct answer', choices))[0]

    @staticmethod
    def from_files(fnames, lang='en'):
        answer_transformations = {}
        total, commented = 0, 0
        for fn in fnames:
            d = json.load(open(fn))
            assert 'data' in d, "Not compatible with Textinator format"

            for dp in d['data']:
                context, question, choices = dp['context'], dp['question'], dp['choices']
                ca = TextinatorLoader.get_correct_answer(choices)

                total += 1
                if ca['extra']:
                    commented += 1
                    answer_transformations[ca['text']] = ca['extra'].get('comment')
                else:
                    answer = {
                        'text': ca['text'],
                        'start': ca['start'],
                        'end': ca['end']
                    }
                    yield question, answer, ca['context']


class SquadLoader:
    @staticmethod
    def collapse_spans(answers):
        answers = sorted(answers, key=lambda x: x['start'])

        N = len(answers)
        checked, collapsed = [False] * N, [False] * N         
        for i in range(1, len(answers)):
            if answers[i-1]['question'] == answers[i]['question'] and answers[i-1]['end'] >= answers[i]['start']:
                # overlap
                N1 = answers[i-1]['end'] - answers[i-1]['start']
                N2 = answers[i]['end'] - answers[i]['start']
                if N1 > N2:
                    collapsed[i-1] = True
                else:
                    collapsed[i] = True
                checked[i-1] = True
                checked[i] = True
            else:
                # no overlap
                if not checked[i-1]:
                    collapsed[i-1] = True
                    checked[i-1] = True
        if not checked[N-1]:
            collapsed[N-1] = True
        return [a for i, a in enumerate(answers) if collapsed[i]]

    @staticmethod
    def from_files(fnames, lang='en'):
        total, total_one_answer = 0, 0
        for fname in fnames:
            info = json.load(open(fname))
            if type(info) == dict:
                assert 'data' in info, "Not compatible with SQuAD format"
                data = info['data']
            elif type(info) == list:
                data = info

            for dp in data:
                for paragraph in dp['paragraphs']:
                    context = paragraph['context']
                    answers = []
                    for qa in paragraph['qas']:
                        if ('is_impossible' not in qa) or (not qa['is_impossible']):# and len(qa['answers']) == 1:
                            total_one_answer += 1
                            answer = qa['answers'][0]
                            answer['question'] = qa['question']
                            answer['start'] = answer['answer_start']
                            answer['end'] = answer['answer_start'] + len(answer['text'])
                            del answer['answer_start']
                            if answer not in answers:
                                answers.append(answer)
                        total += 1

                    if answers:
                        for answer in SquadLoader.collapse_spans(answers):
                            yield answer['question'], answer, context
        print(total, total_one_answer)


class JsonLinesLoader:
    @staticmethod
    def from_files(fnames, lang='en'):
        for fname in fnames:
            with jsonlines.open(fname) as reader:
                for obj in reader:
                    sentence = obj['sentence']
                    question = obj["question"]
                    answer = obj.get("answer", '')
                    yield question, answer, sentence


class TyDiQaLoader:
    @staticmethod
    def from_files(fnames, lang='en'):
        def pretty_stats(dct):
            return "\n".join(["\t{}: {}({}%)".format(k, sdct['v'], round(sdct['v'] * 100 / sdct['t'], 2)) for k, sdct in dct.items()])

        lang2full = {
            'fi': 'finnish',
            'ru': 'russian',
            'en': 'english',
            'ja': 'japanese',
            'te': 'telugu',
            'ar': 'arabic',
            'bn': 'bengali',
            'th': 'thai',
            'id': 'indonesian',
            'ko': 'korean',
            'sw': 'swahili'
        }

        lang = lang2full[lang] # should raise error if language is not present

        total, total_valid, k = 0, 0, 0
        lang_cnt = defaultdict(lambda: {'v': 0, 't': 0})
        for fname in fnames:
            with jsonlines.open(fname) as reader:
                i = 0
                for obj in reader:
                    total += 1
                    lang_cnt[obj['language']]['t'] += 1
                    annotations = obj['annotations']

                    if any([a['yes_no_answer'] != 'NONE' for a in annotations]) or obj['language'] != lang:
                        # skip answers requiring yes or no
                        continue

                    annot_len = len(annotations)
                    if annot_len > 1 and obj['language'] == lang:
                        # all of the languages in the dev set have more than one annotation
                        k += 1

                    doc = obj['document_plaintext'].encode('utf8')
                    candidates = obj['passage_answer_candidates']
                    question = obj['question_text']

                    valid = False
                    for annot in annotations:
                        ca_start = annot['minimal_answer']['plaintext_start_byte']
                        ca_end = annot['minimal_answer']['plaintext_end_byte']
                        passage_index = annot['passage_answer']['candidate_index']

                        if passage_index == -1 or ca_start == 1 or ca_end == 1:
                            continue
                        
                        cand = candidates[passage_index]
                        ca = doc[ca_start:ca_end].decode('utf8', errors="replace")
                        answer = {
                            'text': ca,
                            'start': ca_start - cand['plaintext_start_byte'],
                            'end': ca_end - cand['plaintext_start_byte']
                        }
                        passage = doc[cand['plaintext_start_byte']:cand['plaintext_end_byte']].decode('utf8', errors="replace")
                        total_valid += 1
                        valid = True
                        yield question, answer, passage
                    lang_cnt[obj['language']]['v'] += valid
        # print("Stats:\n{}".format(pretty_stats(lang_cnt)))
        # print("Total valid: {} ({}%)".format(total_valid, round(total_valid / total, 2)))
        # print("More than 1 annotation: {}/{}".format(k, total))


class XquadLoader:
    @staticmethod
    def from_files(fnames, lang='en'):
        total, total_one_answer = 0, 0
        for fname in fnames:
            info = json.load(open(fname))
            assert 'data' in info, "Not compatible with SQuAD format"

            for dp in info['data']:
                for paragraph in dp['paragraphs']:
                    context = paragraph['context']
                    answers = []
                    for qa in paragraph['qas']:
                        if not qa['is_impossible']:# and len(qa['answers']) == 1:
                            total_one_answer += 1
                            answer = qa['answers'][0]
                            answer['question'] = qa['question']
                            answer['start'] = answer['answer_start']
                            answer['end'] = answer['answer_start'] + len(answer['text'])
                            del answer['answer_start']
                            if answer not in answers:
                                answers.append(answer)
                        total += 1

                    if answers:
                        for answer in SquadLoader.collapse_spans(answers):
                            yield answer['question'], answer, context
        print(total, total_one_answer)