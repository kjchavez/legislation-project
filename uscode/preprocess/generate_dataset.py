"""
    Generates training, validation, and test splits of data for U.S. Legal Code
    language model.

    Creates the following files in OUTPUTDIR:

        * vocabulary.txt:  Ordered list of all tokens appearing in the data.
        * train.txt:       Space-separated token ids for training data.
        * valid.txt:       Space-separated token ids for validation data.
        * test.txt:        Space-separated token ids for test data.

"""
import argparse
import os
from vocab import Vocabulary
from tokenizer import tokenize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--data_filepattern", default="uscode_text/usc*.txt")
    parser.add_argument("--max_vocab_size", '-s', type=int, default=10000)
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--min_freq", type=float, default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    vocab = Vocabulary.build(args.data_filepattern, tokenize,
                             args.max_vocab_size,
                             min_freq=args.min_freq,
                             min_count=args.min_count)

    vocab.saveto(os.path.join(args.output_dir, 'vocabulary.txt'))


if __name__ == "__main__":
    main()
