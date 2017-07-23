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

TRANSFORM_FN = {}
TRANSFORM_FN[int] = _int64_feature
TRANSFORM_FN[str] = _bytes_feature
TRANSFORM_FN[unicode] = _bytes_feature
def _transform_function(py_value):
    return TRANSFORM_FN.get(type(py_value))

def _get_schema(example):
    schema = {k : _transform_function(v) for k, v in example.items()}
    if any (fn is None for fn in schema.values()):
        raise Exception("Some transform functions are missing")
    return schema

def to_tf_example(x, schema=None):
    """ Transforms a python dictionary into a TF Example proto. """
    if schema is None:
        schema = _get_schema(example)

    f = {key : schema[key](value) for key, value in x.items()}
    example = tf.train.Example(features=tf.train.Features(feature=f))
    return example


def create_tfrecord(py_examples, filename):
    schema = None
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in py_examples:
            if not schema:
                schema = _get_schema(x)
            example = to_tf_example(x, schema=schema)
            writer.write(example.SerializeToString())

def line_reader(filename, limit=None):
    """ Returns a dict for each line in |filename|. """
    with open(filename) as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                return
            x = eval(line.strip())
            yield x


def get_all_bills(py_examples):
    bills = set()
    for x in py_examples:
        bills.update([x['bill_id']])

    return bills


def partition(x, train=0.7, test=0.2, valid=0.1):
    x = list(x)
    random.shuffle(x)
    N_train = int(train*len(x))
    N_valid = int(test*len(x))
    return x[0:N_train], x[N_train:(N_train+N_valid)], x[(N_train+N_valid):]

def attribute_filters(dataset, attr, train=0.7, valid=0.1, test=0.2):
    """ Returns a set of filter functions for train, valid, test sets.

    Args:
        dataset : iterable of dict
                  Provides access to elements of the dataset
        attr : string
               Key of dict from dataset that should be used to generate filters

    Returns:
        tuple of 3 functions, dict -> bool, which return True if that data element should go in the
        corresponding train/valid/test set.
    """
    attrs = set()
    for x in dataset:
        attrs.update([x.get(attr, None)])

    filters = [lambda x: x.get(attr, None) in s
               for s in partition(attrs, train=train, test=test, valid=valid)]
    return tuple(filters)


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

    logging.info("Partitioning...")
    train, valid, test = attribute_filters(line_reader(args.data), 'bill_id')
    logging.info("...DONE")
    for name, filter_fn in (('train', train), ('valid', valid), ('test', test)):
        out = os.path.join(args.outdir, '%s.tfrecord' % name)
        logging.info("Creating TFRecord @ %s...", out)
        create_tfrecord(itertools.ifilter(filter_fn, line_reader(args.data)), out)
        logging.info("...DONE")


if __name__ == "__main__":
    main()
