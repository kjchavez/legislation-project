"""
    Script for building data sets out of slices of GovTrack bulk congress data.
    This assumes you have downloaded the dataset to the local machine.

    See https://www.govtrack.us/developers/data for details on getting the
    data.

    Sample usage:

        python -m congressdb.build --src /data/congress --output /tmp/data \
                                   --type=s --version=is


    Creates the following files in the |output| directory:

        * vocabulary.txt:  Ordered list of all tokens appearing in the data.
        * train.txt:       Space-separated token ids for training data.
        * valid.txt:       Space-separated token ids for validation data.
        * test.txt:        Space-separated token ids for test data.
        * METADATA:        Additional information about the generated data
"""

import argparse
import shutil
import sys
from preprocessing.vocab import Vocabulary
from preprocessing import constants
from preprocessing import tokenizer
from .congressdb import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--output", default=None)

    # Filters on the data.
    parser.add_argument("--type", choices=['*', 's', 'hr'], default='*')
    parser.add_argument("--version", choices=['*', 'is', 'ih'], default='*')
    parser.add_argument("--congress", type=int, default=None)

    parser.add_argument("--force", '-f', type=bool, default=False,
                        help="If true, will not issue confirmation step to"
                             " overwrite output directory.")

    # Arguments for vocabulary generation.
    parser.add_argument("--prebuilt_vocab", type=str, default=None,
                        help="Vocabulary file to use.")
    parser.add_argument("--max_vocab_size", type=int, default=10000)
    parser.add_argument("--min_count", type=int, default=10)
    parser.add_argument("--min_freq", type=float, default=None)

    return parser.parse_args()


def _prepare_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)

def _check_ok_path(directory, force=False):
    """ Checks that it's okay to overwrite directory. Exits if not. """
    if os.path.exists(directory):
        if not force:
            r = raw_input("Clear dir %s [y/n]?" % directory)
            if r not in ('y', 'Y'):
                sys.exit(1)

def _text_to_token_ids(text, vocab):
    return [vocab.get(t) for t in vocab.tokenize(text)]

def _get_bill_iterator(src, bill_type, bill_version, congress=None):
    cdb = CongressDatabase(src)
    if not congress:
        congress = '*'

    return cdb.bill_text(bill_type=bill_type, version=bill_version,
                         congress_num=congress)


def build_new_vocab(src, bill_type, bill_version, congress=None,
                    max_num_tokens=None, min_freq=None, min_count=None):
    iterator = _get_bill_iterator(src, bill_type=bill_type,
            bill_version=bill_version, congress=congress)
    vocab = Vocabulary.fromiterator(iterator, tokenizer.wordpunctnewline_lower,
                max_num_tokens=max_num_tokens, min_freq=min_freq,
                min_count=min_count,
                extra=[constants.BILL_START, constants.BILL_END])

    return vocab

def _create_default_output_dirname(args):
    dirname = "congress-%s" % (str(args.congress) if args.congress else "all")
    if args.type != '*':
        dirname += '-' + args.type
    if args.version != '*':
        dirname += '-' + args.version
    return dirname

def main():
    args = parse_args()
    if not args.output:
        args.output = _create_default_output_dirname(args)

    _check_ok_path(args.output, force=args.force)
    if args.prebuilt_vocab:
        vocab = Vocabulary.fromfile(args.prebuilt_vocab)
    else:
        vocab = build_new_vocab(args.src, args.type, args.version,
                                congress=args.congress,
                                max_num_tokens=args.max_vocab_size,
                                min_freq=args.min_freq,
                                min_count=args.min_count)

    # Let's not clear the dir until we read the vocab file (since it might be
    # in that directory!)
    _prepare_clean_dir(args.output)
    vocab.saveto(os.path.join(args.output, 'vocabulary.txt'))
    print "Size of vocab:", vocab.size()

    iterator = _get_bill_iterator(args.src, args.type, args.version,
                                  congress=args.congress)

    # TODO(kjchavez): Make these splits configurable and maybe shuffle them
    # For now, default is 80/10/10.
    splits = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    data_files = [os.path.join(args.output, f)
                  for f in ['train.txt', 'validate.txt', 'test.txt']]

    partitions = [[], [], []]
    for idx, (bill_id, bill_text) in enumerate(iterator):
        bucket = splits[idx % len(splits)]
        partitions[bucket].append(bill_id)
        filename = data_files[bucket]
        with open(filename, 'a') as fp:
            print >> fp, vocab.get(constants.BILL_START)
            for idx in _text_to_token_ids(bill_text, vocab):
                print >> fp, idx
            print >>fp, vocab.get(constants.BILL_END)

    with open(os.path.join(args.output, "METADATA"), 'w') as fp:
        for i in xrange(len(data_files)):
            print >> fp, "Ids for %s:" % os.path.basename(data_files[i])
            print >> fp, '\n'.join(partitions[i])

        print >> fp, "\nArgs for build.py:"
        print >> fp, args

if __name__ == "__main__":
    main()
