import argparse
import collections
import itertools
import json
import logging
import os
import random
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))


def tf_example_from_dict(x):
    sponsor_party = x['bill'].get('sponsor_party', 'unknown')
    member_party = x['member'].get('most_recent_party', 'unknown')
    voted_aye = x['aye']
    example = tf.train.Example(features=tf.train.Features(feature={
        'member_party': _bytes_feature(member_party),
        'sponsor_party': _bytes_feature(sponsor_party),
        'voted_aye': _int64_feature(voted_aye)
    }))
    return example


def create_tfrecord(py_examples, filename):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in py_examples:
            example = tf_example_from_dict(x)
            writer.write(example.SerializeToString())

def line_reader(filename, limit=None):
    """ Returns a dict for each line in |filename|. """
    with open(filename) as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                return
            x = eval(line.strip())
            yield x


def sponsor_member_party(x):
    sponsor_party = x['bill'].get('sponsor_party', 'unknown')
    member_party = x['member'].get('most_recent_party', 'unknown')
    return (sponsor_party, member_party)


def count_party_alignment(datafile):
    counter = collections.Counter()
    counter.update(sponsor_member_party(x) for x in line_reader(datafile))
    for k, v in counter.items():
        print k, v


def get_all_bills(py_examples):
    bills = set()
    for x in py_examples:
        bills.update([x['bill']['id']])

    return bills


def partition(x, train=0.7, test=0.2, valid=0.1):
    random.shuffle(x)
    N_train = int(train*len(x))
    N_valid = int(test*len(x))
    return x[0:N_train], x[N_train:(N_train+N_valid)], x[(N_train+N_valid):]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/votes.dat")
    parser.add_argument("--outdir", '-o', default='data')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    logging.info("Finding all bill ids...")
    bill_ids = list(get_all_bills(line_reader(args.data)))
    logging.info("Partitioning...")
    train, valid, test = partition(bill_ids)
    logging.info("...DONE")
    for name, ids in (('train', train), ('valid', valid), ('test', test)):
        out = os.path.join(args.outdir, '%s.tfrecord' % name)
        logging.info("Creating TFRecord @ %s...", out)
        create_tfrecord(itertools.ifilter(lambda x: x['bill']['id'] in ids, line_reader(args.data)), out)
        logging.info("...DONE")


if __name__ == "__main__":
    main()
