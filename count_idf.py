import csv
import math
import stanza
import argparse
from collections import defaultdict

from tqdm import tqdm

import loaders



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='A language for template generation (en, sv are currently supported)')
    parser.add_argument('-f', '--files', type=str, help='Comma-separated list of files to generate questions from')
    parser.add_argument('-o', '--out-file', type=str, default='', help='Name of the output file')
    parser.add_argument('-cf', '--case-folding', action='store_true')
    parser.add_argument('-ft', '--format', type=str, help='Data format (tt for Textinator or squad for Squad)')
    parser.add_argument('-lv', '--level', type=str, default='doc', help='Either "doc" or "sent"')
    args = parser.parse_args()

    if args.format == 'tt':
        data_loader = loaders.TextinatorLoader
    elif args.format == 'squad':
        data_loader = loaders.SquadLoader
    elif args.format == 'tydiqa':
        data_loader = loaders.TyDiQaLoader

    stanza_pipe = stanza.Pipeline(lang=args.lang, processors='tokenize,mwt,pos' if args.lang in ['fi', 'ar'] else 'tokenize,pos')

    N, df = 0, defaultdict(int)
    if args.level == 'doc':
        for _, _, c in tqdm(data_loader.from_files(args.files.split(','), args.lang)):
            doc = stanza_pipe(c.lower() if args.case_folding else c)
            words = set()
            for sent in doc.sentences:
                for word in sent.tokens:
                    words.add(word.text)

            for w in words:
                df[w] += 1
            N += 1
    elif args.level == 'sent':
        for _, _, c in tqdm(data_loader.from_files(args.files.split(','), args.lang)):
            doc = stanza_pipe(c.lower() if args.case_folding else c)
            for sent in doc.sentences:
                for word in set(sent.tokens):
                    df[word.text] += 1
                N += 1

    with open(args.out_file or 'idf_{}.csv'.format(args.lang), 'w') as f:
        writer = csv.writer(f)
        for w in df:
            writer.writerow([w, math.log(N / df[w])])