import argparse
from collections import defaultdict
from pprint import pprint

import dill
import stanza
from tqdm import tqdm

from quinductor.loaders import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='A language for stats generation (en, sv are currently supported)')
    parser.add_argument('-d', '--data', type=str, help='Comma-separated list of files to generate stats from')
    parser.add_argument('-ft', '--format', type=str, help='Data format (tt for Textinator or squad for Squad)')
    parser.add_argument('-rtl', '--right-to-left', action='store_true')
    args = parser.parse_args()

    questions_fnames = [x.strip() for x in args.data.split(',')]

    # arabic, finnish - include mwt
    # russian - exclude mwt
    stanza_processors = 'tokenize,lemma,mwt,pos,depparse' if args.lang in ['fi', 'ar'] else 'tokenize,lemma,pos,depparse'
    stanza_lang = stanza.Pipeline(lang=args.lang, processors=stanza_processors)

    if args.format == 'tt':
        data_loader = TextinatorLoader
    elif args.format == 'squad':
        data_loader = SquadLoader
    elif args.format == 'tydiqa':
        data_loader = TyDiQaLoader

    stats = defaultdict(lambda: defaultdict(int))
    answer_tmpl = defaultdict(int)
    for q, a, c in tqdm(data_loader.from_files(questions_fnames, args.lang)):
        q_parsed = stanza_lang(q)
        qw = q_parsed.sentences[0].words[-1].text if args.right_to_left else q_parsed.sentences[0].words[0].text
        if not a['text']: continue
        a_parsed = stanza_lang(a['text'])

        aw = None
        for w in a_parsed.sentences[0].words:
            if w.deprel == 'root':
                aw = w
                break

        if aw.feats:
            morph = '|'.join(sorted(aw.feats.split('|')))
        else:
            morph = None

        at = aw.upos + '/' + morph if morph else aw.upos
        stats[qw.lower()][at] += 1

        answer_tmpl[" ".join(
            [x.upos + '/' + '|'.join(sorted(x.feats.split('|'))) if x.feats else x.upos
            for x in a_parsed.sentences[0].words])] += 1

    dill.dump(stats, open('qwstats.dill', 'wb'))
    dill.dump(answer_tmpl, open('atmpl.dill', 'wb'))

