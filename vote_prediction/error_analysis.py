# Collect sample errors on a dev set using a SavedModel.
# Note, this is well suited for a Jupyter Notebook.
# Hint, hint, future self.
from __future__ import print_function

import argparse
import numpy as np
import logging
import os
import json
import itertools
import tensorflow as tf


def _collect_batch_errors(batch_output, batch_examples, collect_errors_fn):
    results = []
    batch_size = len(batch_output.values()[0])
    for i in xrange(batch_size):
        output = dict((key, batch_output[key][i]) for key in batch_output.keys())
        example = dict((key, batch_examples[key][i]) for key in batch_examples.keys())
        tags = collect_errors_fn(output, example)
        if tags:
            results.append({'example': example, 'errors' : tags})

    return results


# Turn this into a bit of a framework.
class ErrorAnalysis(object):
    def __init__(self, error_tags, batch_generator):
        """
        Args:
            error_tags: dict of string to function: X, Y -> bool. Function should determine if
                        response Y on example X is an error of that type.
            batch_generator: yields a dictionary with batched features/label on each iteration.
        """
        self.error_tags = error_tags
        self.batch_generator = batch_generator

    def collect_errors(self, output, example):
        tags = []
        for tag, func in self.error_tags.items():
            if func(example, output):
                tags.append(tag)
        return tags

    def analyze(self, model_dir, outfile):
        if os.path.exists(outfile):
            logging.info("Clearing file: %s", outfile)
            with open(outfile, 'w') as fp:
                pass

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            metagraph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                      model_dir)
            signature_def = metagraph_def.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

            print("Default SIGNATURE_DEF:")
            print("="*80)
            print(signature_def)
            print("="*80)

            # In generating the batch data, we may need to apply some kind of transformation
            # to the dev data set -- it may not be in the same format as inference time.
            #
            # I can really only see one way to get in this situation
            #
            # 1. You want the test set to be representative of "inference" time data distribution.
            # 2. So the input format should probably be the same.
            # 3. The dev set should be from the same distribution as the test set.
            # 4. So input transformations on dev should == test == inference.
            # 5. But we evaluate on the dev set frequently, so we may want to "cache" some of that
            #    preprocessing. Fine. But be careful to keep it aligned.
            for batch in self.batch_generator:
                feed = {}
                for name, tensor_info in signature_def.inputs.items():
                    feed[tensor_info.name] = batch[name]

                outputs = {}
                for name, tensor_info in signature_def.outputs.items():
                    outputs[name] = tensor_info.name

                output_val = sess.run(outputs, feed_dict=feed)
                with open(outfile, 'a') as fp:
                    for x in _collect_batch_errors(output_val, batch,
                                                   self.collect_errors):
                        print(json.dumps(x), file=fp)


def take(iterable, n):
    return list(itertools.islice(iterable, n))


def batch_generator(example_iter, batch_size):
    while True:
        examples = take(example_iter, batch_size)
        if not examples:
            return
        batch = dict((key, []) for key in examples[0].keys())
        for x in examples:
            for key in x.keys():
                batch[key].append(x[key])
        # Do we need to convert to numpy arrays?
        yield batch


def batch_generator_from_file(filename, batch_size):
    def eval_line_generator():
        with open(filename) as fp:
            for line in fp:
                yield eval(line)

    return batch_generator(eval_line_generator(), batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Saved model directory.")
    parser.add_argument("--examples", required=True, help="TFRecord with examples.")
    parser.add_argument("--outfile", default="errors.dat")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def pred_error(x, y):
    return bool(y['aye']) != bool(x['Decision'] == 'Aye')

def main():
    args = parse_args()
    err = ErrorAnalysis({'prediction_error': pred_error}, batch_generator_from_file(args.examples,
                                                                                    args.batch_size))
    err.analyze(args.model, args.outfile)


if __name__ == "__main__":
    main()
