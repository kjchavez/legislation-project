import argparse
import collections
import json
import logging
import os
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='data/votes.train.dat')
    parser.add_argument("--valid", default='data/votes.valid.dat')
    parser.add_argument("--test", default='data/votes.test.dat')
    parser.add_argument("--outdir", '-o', default='data')
    return parser.parse_args()


def main():
    args = parse_args()
    for f in (args.train, args.valid, args.test):
        if not os.path.exists(f):
            logging.warning("File %s does not exist.", f)
            continue
        out = os.path.splitext(f)[0]+'.tfrecord'
        create_tfrecord(line_reader(f), out)
        logging.info("Created TFRecord @ %s", out)

if __name__ == "__main__":
    main()
