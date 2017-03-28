import argparse
import collections
import constants
import glob
import tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepattern", '-f', default="uscode_text/*.txt")
    parser.add_argument("--vocab_file", '-v', default="vocab.txt")
    parser.add_argument("--max_vocab_size", '-s', type=int, default=5000)
    parser.add_argument("--min_freq", '-m', type=float, default=1e-6)

    return parser.parse_args()


def freq(key, counter):
    return float(counter[key]) / sum(counter.values())

def create_vocab_file(counter, filename, max_vocab_size=1000, min_freq=1e-6):
    with open(filename, 'w') as fp:
        print >> fp, constants.END_OF_FILE
        for key, count in counter.most_common(max_vocab_size):
            if freq(key, counter) < min_freq:
                print "Exiting early because word frequency is below", min_freq
                break

            print >> fp, key.encode('utf8'), count

def main():
    counter = collections.Counter()
    args = parse_args()
    for filename in glob.glob(args.filepattern):
        print "Processing:", filename
        with open(filename) as fp:
            for line in fp:
                counter.update(tokenize.tokenize(line))

    create_vocab_file(counter, args.vocab_file,
                      max_vocab_size=args.max_vocab_size,
                      min_freq=args.min_freq)


if __name__ == "__main__":
    main()

