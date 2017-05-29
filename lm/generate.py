"""
    Generates a sample using the language model in a given checkpoint.

    Sample usage:

        python generate.py --model_dir /tmp/models --length 200
"""
import argparse
import tensorflow as tf
import numpy as np
import logging
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
    parser.add_argument("--temp", '-T', type=float, default=1.0)
    parser.add_argument("--hyperparams", default=None,
                        help="yaml file of hyperparameters")
    parser.add_argument("--max_length", type=int, default=1000,
                        help="Max number of tokens to generate.")
    parser.add_argument("--output", '-o', type=str, default=None)

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

    token_ph = tf.placeholder(dtype=tf.int32, shape=(params['batch_size'],
                                                     params['unroll_length']))
    features = {'tokens': token_ph}

    with tf.name_scope("Train"):
      with tf.variable_scope("Model"):
        model = LanguageModel(features=features, targets=None,
                    mode="GENERATE", params=params)

    sv = tf.train.Supervisor(logdir=args.model_dir)
    with sv.managed_session() as session:
        token_ids = session.run(model.output_tokens, feed_dict =
                {model.temperature: args.temp})[0]

    print "Generated sample of length %d" % len(token_ids)
    if not args.output:
        print ' '.join(vocab.get_token(idx) for idx in token_ids)
    else:
        with open(args.output, 'w') as fp:
            fp.write(' '.join(vocab.get_token(idx) for idx in token_ids))

if __name__ == "__main__":
    main()
