# Collect sample errors on a dev set using a SavedModel.
import argparse
import numpy as np
import logging
import os
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Saved model directory.")
    parser.add_argument("--examples", required=True, help="TFRecord with examples.")
    parser.add_argument("--outfile", default="errors.dat")
    return parser.parse_args()

def collect_errors(output, example):
    """ Returns tags for all types of errors made by the model on this example."""
    logging.debug("Evaluating %s", output)
    tags = []
    if bool(output['aye']) != bool(example['aye']):
        tags.append('predict_error')

    return tags

def collect_batch_errors(batch_output, batch_examples):
    results = []
    batch_size = len(batch_output.values()[0])
    for i in xrange(batch_size):
        output = dict((key, batch_output[key][i]) for key in batch_output.keys())
        example = dict((key, batch_examples[key][i]) for key in batch_examples.keys())
        tags = collect_errors(output, example)
        if tags:
            results.append({'example': example, 'errors' : tags})

    return results


def main():
    args = parse_args()

    if os.path.exists(args.outfile):
        with open(args.outfile, 'w') as fp:
            pass

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        metagraph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model)
        signature_def = metagraph_def.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        print("Default SIGNATURE_DEF:")
        print("="*80)
        print(signature_def)
        print("="*80)
        # |batch| should be a dictionary containing a batch of Examples.
        # Fake batch example.
        batch = {'VoterAge': [25, 26],
                 "VoterState": ["MA", "CA"],
                 "VoterParty": ["democrat", "republican"],
                 "SponsorParty": ["democrat", "democrat"],
                 "BillTitle": ["Welcome to the jungle!", "Welcome to the jungle!"],
                 'aye': [0, 1]}

        feed = {}
        for name, tensor_info in signature_def.inputs.items():
            print(name)
            feed[tensor_info.name] = batch[name]

        outputs = {}
        for name, tensor_info in signature_def.outputs.items():
            outputs[name] = tensor_info.name

        output_val = sess.run(outputs, feed_dict=feed)
        with open(args.outfile, 'a') as fp:
            for x in collect_batch_errors(output_val, batch):
                print >> fp, x


if __name__ == "__main__":
    main()
