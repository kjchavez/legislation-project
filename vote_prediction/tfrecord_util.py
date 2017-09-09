import argparse
import collections
import itertools
import json
import logging
import os
import random
import itertools
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(value)]))

TRANSFORM_FN = {}
TRANSFORM_FN[int] = _int64_feature
TRANSFORM_FN[str] = _bytes_feature
TRANSFORM_FN[unicode] = _bytes_feature
def _transform_function(py_value):
    if type(py_value) == list and type(py_value[0]) == int:
        return _int64_list_feature

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


def transform_example(x, tokenizer=None, text_fields=[]):
    if tokenizer and text_fields:
        for field in text_fields:
            if field in x:
                x[field] = tokenizer.texts_to_sequences([str(x[field])])[0]
    return x


def partition(x, train=0.7, valid=0.1, test=0.2):
    x = list(x)
    random.shuffle(x)
    N_train = int(train*len(x))
    N_valid = int(valid*len(x))
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

    parts = partition(attrs, train=train, valid=valid, test=test)
    for p in parts:
        print "== PARTITITION =="
        print "Length:", len(p)
        print "Fingerprint:", sum(hash(i) % int(1e6) for i in p)

    # A for loop is tempting, but tricky here. Remember, in Python, assignment
    # only makes a new reference to the same underlying object -- and lambdas don't have the best
    # capture syntax. So you can easily end up with all functions referring to the same partition.
    filters = []
    filters.append(lambda x: x.get(attr) in parts[0])
    filters.append(lambda x: x.get(attr) in parts[1])
    filters.append(lambda x: x.get(attr) in parts[2])
    return tuple(filters)


# TODO(kjchavez): move the Vocabulary() class from preprocessing directory
# into shared util python package that can be shared among all these models.
def tokenize(text):
    """ Trivial tokenization implementation. """
    return text.split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/votes.dat")
    parser.add_argument("--text_fields", type=str, nargs='+',
                        help="names of text fields that should be used to create vocab.")
    parser.add_argument("--outdir", '-o', default='data')
    return parser.parse_args()


def tokens_from_iterator(iterator, tokenize_fn,
                 max_num_tokens=None, min_freq=None, min_count=None,
                 extra=[]):
    """ Builds vocab from an iterator that yields (id, text) tuples."""
    counter = collections.Counter()
    for idx, text in iterator:
        counter.update(tokenize_fn(text))

    tokens = extra
    for key, count in counter.most_common(max_num_tokens):
        if (min_freq and _freq(key, counter) < min_freq) or \
           (min_count and count < min_count):
            break
        tokens.append(key)

    return tokens

def text_field_generator(examples, text_fields):
    for x in examples:
        for field in text_fields:
            if field in x.keys():
                yield str(x[field])

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    logging.info("Partitioning...")
    train, valid, test = attribute_filters(line_reader(args.data), 'BillId')
    logging.info("...DONE")
    tokenizer = None
    if args.text_fields:
        vocab = collections.Counter()
        logging.info("Creating vocabulary from text fields.")
        tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer()
        it = text_field_generator(itertools.ifilter(train, line_reader(args.data)),
                                  args.text_fields)
        tokenizer.fit_on_texts(it)
        with open(os.path.join(args.outdir, 'vocab.txt'), 'w') as fp:
            print >> fp, "UNK"
            words = sorted([(idx, token) for token, idx in tokenizer.word_index.items()])
            for word in words:
                print >> fp, word

        logging.info("...DONE")

    transform = lambda x: transform_example(x, tokenizer=tokenizer,
                                    text_fields=args.text_fields)
    for name, filter_fn in (('train', train), ('valid', valid), ('test', test)):
        out = os.path.join(args.outdir, '%s.tfrecord' % name)
        logging.info("Creating TFRecord @ %s...", out)
        create_tfrecord(itertools.imap(transform, itertools.ifilter(filter_fn,
                                                                    line_reader(args.data))), out)
        logging.info("...DONE")


if __name__ == "__main__":
    main()
