import argparse
import tqdm

import stanza

from common import get_intersection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='Language of interest')
    parser.add_argument('-f', '--files', type=str, help='Comma-separated list of QAC files')
    args = parser.parse_args()

    proc = 'tokenize,mwt,pos' if args.lang in ['fi', 'ar'] else 'tokenize,pos'
    stanza_tokenizer = stanza.Pipeline(lang=args.lang, processors=proc)

    possible, uses_constituents, q_possible, total = 0, 0, 0, 0
    for fname in args.files.split(','):
        with open(fname) as f:
            for line in tqdm.tqdm(f):
                if line.strip():
                    # base_sentence is pretokenized so that tokens are separated by spaces
                    question, answer, base_sentence = line.split(" #|@ ")

                    if not question or not answer or not base_sentence:
                        continue

                    ss = stanza_tokenizer(base_sentence)
                    s_tokens = [w.text for s in ss.sentences for w in s.words]

                    aa = stanza_tokenizer(answer)
                    a_tokens = [w.text for s in aa.sentences for w in s.words]

                    qq = stanza_tokenizer(question)
                    q_tokens = [w.text for s in qq.sentences for w in s.words]

                    same = get_intersection(q_tokens, s_tokens)
                    cond = len(same) > 0

                    q_possible += cond

                    if (set(a_tokens) & set(s_tokens)) != set(a_tokens):
                        print(a_tokens, s_tokens)
                        uses_constituents += 1
                    else:
                        possible += cond
                    total += 1
    print("Uses constituents: {} ({}%). Total: {}.".format(uses_constituents, round(uses_constituents / total, 4), total))
    print("Question possible: {} ({}%) and not uses constituents: {} ({}%).".format(
        q_possible, round(q_possible / total, 4), possible, round(possible / total, 4)))
