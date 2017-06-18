"""
    Computes perplexity on the test set.

    Sample usage:

        python evaluate.py --model_dir /tmp/models \
                           --data_path=house-introduced-114
                           --hparams=hparams.yaml
"""
import argparse
import tensorflow as tf
import numpy as np
import logging
import yaml

from lm.model import LanguageModel, InputData, run_epoch
from lm.sample import sample
from lm.reader import load_vocab
from lm import reader

OOV_ID = 0
SOB_TOKEN_ID = 1
EOB_TOKEN_ID = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/tmp/house-model")
    parser.add_argument("--data_path", default="house-introduced-114")
    parser.add_argument("--hparams", default=None, required=True,
                        help="yaml file of hyperparameters")

    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab(args.data_path)

    if args.hparams:
        with open(args.hparams) as fp:
            params = yaml.load(fp)

    params['vocab_size'] = vocab.size()
    params['vocab'] = vocab.ordered_tokens()

    # We could, in principle, generate a batch of random samples. For
    # simplicity we won't.
    params['batch_size'] = 1

    # This is so we can extract the probability distribution over tokens after
    # a single step and employ a sampling strategy of our choosing.
    params['unroll_length'] = 1

    dataset = reader.LegislationDataset(args.data_path)
    vocab_data = reader.load_vocab(args.data_path)
    with tf.name_scope("Test"):
      test_batches = dataset.test_batch_generator(1, 1)
      test_input = InputData(params=params,
                             batched_dataset=test_batches,
                             name="TestInput")
      with tf.variable_scope("Model"):
        mtest = LanguageModel(mode=tf.contrib.learn.ModeKeys.EVAL, params=params,
                         features=test_input.input_data,
                         targets=test_input.targets)

    sv = tf.train.Supervisor(logdir=args.model_dir)
    with sv.managed_session() as session:
        test_input.start_queue_thread(session)
        test_perplexity = run_epoch(session, mtest, verbose=True,
                epoch_size=test_batches.epoch_size)
        print ("Test set perplexity: %.3f" % test_perplexity)
        test_input.shutdown(session)

if __name__ == "__main__":
    main()
