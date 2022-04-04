import csv
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='A sentences file to get statistics from')
    args = parser.parse_args()

    support = []
    with open(args.file) as f:
        for line in f:
            if line.startswith("id: "):
                support.append(0)
            elif line.strip():
                support[-1] += 1

    print("Support")
    print("\t MEAN (SD): {} ({})".format(np.mean(support), np.std(support)))
    print("\t MEDIAN (MIN - MAX): {} ({} - {})".format(np.median(support), np.min(support), np.max(support)))
