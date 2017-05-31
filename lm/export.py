"""
    Generates a sample using the language model in a given checkpoint.

    Sample usage:

        python generate.py --model_dir /tmp/models --length 200
"""
import argparse
import tensorflow as tf
import numpy as np
import logging
import os
from lm.model import LanguageModel
from sample import sample
from reader import load_vocab
import yaml

OOV_ID = 0
SOB_TOKEN_ID = 1
EOB_TOKEN_ID = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/tmp/house-model")
    parser.add_argument("--data_path",
        default="house-introduced-114")
    parser.add_argument("--hyperparams", default=None,
                        help="yaml file of hyperparameters")
    parser.add_argument("--export_dir", help="SavedModelBuilder export dir.",
                        default="/tmp/serve/house-model")
    parser.add_argument("--version", type=int, default=1)

    return parser.parse_args()

def main():
    args = parse_args()
    vocab = load_vocab(args.data_path)

    if args.hyperparams:
        with open(args.hyperparams) as fp:
            params = yaml.load(fp)

    params['vocab_size'] = vocab.size()

    # We could, in principle, generate a batch of random samples. For
    # simplicity we won't.
    params['batch_size'] = 1

    # This is so we can extract the probability distribution over tokens after
    # a single step and employ a sampling strategy of our choosing.
    params['unroll_length'] = 1
    params['vocab'] = vocab.ordered_tokens()

    token_ph = tf.placeholder(dtype=tf.int32, shape=(params['batch_size'],
                                                     params['unroll_length']))
    features = {'tokens': token_ph}

    builder = tf.saved_model.builder.SavedModelBuilder(
                os.path.join(args.export_dir, str(args.version)))

    with tf.name_scope("Train"):
      with tf.variable_scope("Model"):
        model = LanguageModel(features=features, targets=None,
                    mode="GENERATE", params=params)

    sv = tf.train.Supervisor(logdir=args.model_dir)
    with sv.managed_session() as session:
      session.graph._unsafe_unfinalize()
      generate_signature = \
        tf.saved_model.signature_def_utils.build_signature_def(
                {'temp' :
                    tf.saved_model.utils.build_tensor_info(model.temperature)},
              {'tokens':
                  tf.saved_model.utils.build_tensor_info(model.output_tokens)},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
      builder.add_meta_graph_and_variables(session,
          [tf.saved_model.tag_constants.SERVING],
          legacy_init_op=tf.tables_initializer(),
          signature_def_map={
              "GenerateSample": generate_signature,
          })


    builder.save()
    print("Exported.")

if __name__ == "__main__":
    main()
