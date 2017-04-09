"""
    Generates training, validation, and test splits of data for U.S. Legal Code
    language model.

    Creates the following files in OUTPUTDIR:

        * vocabulary.txt:  Ordered list of all tokens appearing in the data.
        * train.txt:       Space-separated token ids for training data.
        * valid.txt:       Space-separated token ids for validation data.
        * test.txt:        Space-separated token ids for test data.
        * METADATA:        Additional information about the generated data

"""
import argparse
import glob
import os
import random
from vocab import Vocabulary
from tokenizer import tokenize

# For testing fast tokenization
# def tokenize(text):
#     return text.split()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--force", '-f', type=bool, default=False,
                        help="If true, will not issue confirmation step to"
                             " overwrite output directory.")
    parser.add_argument("--data_filepattern", default="uscode_text/usc*.txt")
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--min_freq", type=float, default=None)

    return parser.parse_args()


def _get_file_splits(filepattern, valid=0.2, test=0.1):
    """ Returns random train/valid/test splits at the file level. """
    all_files = glob.glob(filepattern)
    assert len(all_files) >= 3

    valid_idx = max(int(valid*len(all_files)), 1)
    test_idx = valid_idx + max(int(test*len(all_files)), 1)

    random.shuffle(all_files)
    valid_filenames = all_files[0:valid_idx]
    test_filenames = all_files[valid_idx:test_idx]
    train_filenames = all_files[test_idx:]
    return train_filenames, valid_filenames, test_filenames

def _files_to_token_ids(filenames, vocab):
    token_ids = []
    for filename in filenames:
        with open(filename) as fp:
            for line in fp:
                token_ids.extend([vocab.get(t) for t in tokenize(line)])

    return token_ids

def _write_ids_to_file(token_ids, filename):
    with open(filename, 'w') as fp:
        for idx in token_ids:
            print >> fp, idx

class Directory(object):
    def __init__(self, name):
        self.name = name

    def file(self, filename):
        return os.path.join(self.name, filename)

def main():
    args = parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        r = raw_input("Output directory <%s> already exists. Proceed anyway "
                      "[y/n]?" % args.output_dir)
        if r not in ('y', 'Y'):
            return

    if not os.path.isdir(args.output_dir):
        print "Creating output directory <%s>." % args.output_dir
        os.makedirs(args.output_dir)

    outdir = Directory(args.output_dir)

    vocab = Vocabulary.build(args.data_filepattern, tokenize,
                             args.max_vocab_size,
                             min_freq=args.min_freq,
                             min_count=args.min_count)

    vocab.saveto(os.path.join(args.output_dir, 'vocabulary.txt'))

    train_files, valid_files, test_files = \
        _get_file_splits(args.data_filepattern)

    # Training data
    train_token_ids = _files_to_token_ids(train_files, vocab)
    _write_ids_to_file(train_token_ids, outdir.file('train.txt'))

    # Validation data
    valid_token_ids = _files_to_token_ids(valid_files, vocab)
    _write_ids_to_file(valid_token_ids, outdir.file('valid.txt'))

    # Test data
    test_token_ids = _files_to_token_ids(test_files, vocab)
    _write_ids_to_file(test_token_ids, outdir.file('test.txt'))

    with open(outdir.file('METADATA'), 'w') as fp:
        print >> fp, "Training files:"
        print >> fp, '\n'.join(train_files)
        print >> fp, "\nValidation files:"
        print >> fp, '\n'.join(valid_files)
        print >> fp, "\nTest files:"
        print >> fp, '\n'.join(test_files)
        print >> fp, "\nArgs for generate_dataset.py:"
        print >> fp, args

if __name__ == "__main__":
    main()
