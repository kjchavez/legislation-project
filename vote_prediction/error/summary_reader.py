""" Extracts error examples from Event summaries saved during training/evaluation. """
import argparse
import glob
import logging
import numpy as np
import os
import tensorflow as tf


def read_batch_tensors(event_filepattern, namescope="error_analysis", tag_set=None):
    """ Yields dictionaries of numpy arrays from saved events.

    Args:
        event_filepattern: glob expression for event files to process
        namescope: extracts tensor summaries created in this name scope.
        tag_set: if set, restricts to only tensors with a tag in this set.

    Returns:
        iterator over dict of numpy arrays with values of the tensors
    """
    filenames = sorted(glob.glob(event_filepattern))
    if not filenames:
        logging.warning("No files match pattern: %s", event_filepattern)

    for filename in filenames:
        logging.info("Processing file: %s", filename)
        for e in tf.train.summary_iterator(filename):
            if not e.HasField('summary'):
                logging.debug("Skipping event without summary.")
                continue

            x = {}
            for v in e.summary.value:
                # I believe TensorFlow V1.2 doesn't populate the tag for tensor_summary ops.
                tag = v.tag if v.tag else v.node_name
                logging.info(tag)
                if tag.startswith(namescope) and v.HasField('tensor'):
                    tag = os.path.basename(tag)
                    if not tag_set or tag in tag_set:
                        x[tag] = tf.make_ndarray(v.tensor)
            yield x


def _unbatch(batch_iterator):
    """ Slices elements of |batch_iterator| along the 0th axis, yielding
        one at a time.
    """
    for batch in batch_iterator:
        for idx in xrange(len(batch.values()[0])):
            yield dict((k, batch[k][idx]) for k in batch.keys())


def read_tensors(event_filepattern, namescope="error_analysis", tag_set=None):
    """ Returns iterator over single example at a time. """
    return _unbatch(read_batch_tensors(event_filepattern, namescope=namescope, tag_set=tag_set))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, help="filepattern for event files")
    return parser.parse_args()

def main():
    args = parse_args()
    for x in read_tensors(args.events):
        print(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
