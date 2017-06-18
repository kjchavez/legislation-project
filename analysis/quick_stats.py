import argparse
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Data directory")
    return parser.parse_args()

def _get_lengths(start_token, end_token, ordered_tokens):
    lengths = []
    prev_start = None
    for i, tk in enumerate(ordered_tokens):
        if tk == start_token:
            prev_start = i
        elif tk == end_token:
            if prev_start is None:
                raise Exception("Malformed tokens")
            lengths.append(i - prev_start)
            prev_start = None

    return lengths

def _all_lengths(data_path):

    with open(os.path.join(data_path, "train.txt")) as train, \
         open(os.path.join(data_path, "validate.txt")) as valid, \
         open(os.path.join(data_path, "test.txt")) as test:
        tokens = itertools.chain([l.strip() for l in train],
                                 [l.strip() for l in valid],
                                 [l.strip() for l in test])
        lengths = _get_lengths('1', '2', tokens)

    return lengths


def main():
    args = parse_args()
    print "Computing bill lengths..."
    lengths = _all_lengths(args.data_path)
    print "== Stats =="
    print "Average length: %0.1f" % np.mean(lengths)
    print "Std dev: %0.1f" % np.std(lengths)

    fig = plt.figure()
    # Note this histogram is a going to be a bit skewed by some of the *really*
    # long bills. See
    # http://www.politico.com/blogs/on-congress/2009/11/gop-wrote-5-of-10-longest-bills-023067
    lengths.sort()
    y = np.linspace(0, 1, len(lengths))
    plt.plot(lengths, y)
    plt.xlabel('Num Tokens')
    plt.ylabel('Percentile')
    plt.title('Bill lengths')
    plt.axis([0, 50000, 0.0, 1.0])
    plt.grid(True)
    plt.savefig("bill_length.png")

if __name__ == "__main__":
    main()
