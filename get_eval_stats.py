import csv
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='CSV file to get statistics from')
    args = parser.parse_args()

    with open(args.file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')

        header = next(reader)
        q_per_sent = [0]
        for row in reader:
            if row:
                q_per_sent[-1] += 1
            else:
                q_per_sent.append(0)
        q_per_sent = q_per_sent[:-1] # the last 0 is due to CSV formatting
        print("Total generated questions: {}".format(sum(q_per_sent)))
        print("Total cases with induced templates: {}".format(len(q_per_sent)))
        print("Cases with all templates filtered out (basic): {}".format(q_per_sent.count(0)))
        print("\t MEAN (SD): {} ({})".format(np.mean(q_per_sent), np.std(q_per_sent)))
        print("\t MEDIAN (MIN - MAX): {} ({} - {})".format(np.median(q_per_sent), np.min(q_per_sent), np.max(q_per_sent)))
