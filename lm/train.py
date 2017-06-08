from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf
import yaml
import datetime
import argparse

from lm.model import InputData, LanguageModel, run_epoch
import lm.reader as reader

ModeKeys = tf.contrib.learn.ModeKeys
tf.logging.set_verbosity(tf.logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', help='YAML file of hyperparameters')
    parser.add_argument('--data_path', help="Where data files are stored.",
                        default="house-introduced-114")
    parser.add_argument('--model_dir', help="to save model snapshots",
                        default="/tmp/house-model")
    parser.add_argument("--reset", action='store_true',
                        help="If true, clears any data in 'model_dir' ")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    return parser.parse_args()

def get_params(hparams_file):
    with open(hparams_file) as fp:
        return yaml.load(fp)

def save_params(params, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = 'params.{:%Y-%m-%d.%H:%M:%S}.yaml'.format(datetime.datetime.now())
    with open(os.path.join(model_dir, filename), 'w') as fp:
        yaml.dump(params, fp, default_flow_style=False)

def main():
  args = parse_args()
  if not args.data_path:
    raise ValueError("Must set --data_path to data directory")

  if args.reset and os.path.exists(args.model_dir):
	shutil.rmtree(args.model_dir)

  dataset = reader.LegislationDataset(args.data_path)

  params = get_params(args.hparams)
  # TODO(kjchavez): A nice way for the model to declare required hyper params.

  save_params(params, args.model_dir)
  vocab = reader.load_vocab(args.data_path)
  params['vocab'] = vocab.ordered_tokens()

  eval_params = get_params(args.hparams)
  eval_params['batch_size'] = 1
  eval_params['unroll_length'] = 1
  eval_params['vocab'] = vocab.ordered_tokens()

  with tf.Graph().as_default():
    # TODO(kjchavez): This should move into the model?
    initializer = tf.random_uniform_initializer(-params['init_scale'],
                                                params['init_scale'])

    with tf.name_scope("Train"):
      train_batches = dataset.train_batch_generator(params['batch_size'],
                                                    params['unroll_length'])
      train_input = InputData(params=params,
                              batched_dataset=train_batches,
                              name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = LanguageModel(mode=ModeKeys.TRAIN, params=params,
                features=train_input.input_data, targets=train_input.targets)

    with tf.name_scope("Valid"):
      valid_batches = dataset.valid_batch_generator(params['batch_size'],
                                                    params['unroll_length'])
      valid_input = InputData(params=params,
                              batched_dataset=valid_batches,
                              name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = LanguageModel(mode=ModeKeys.EVAL, params=params,
                features=valid_input.input_data, targets=valid_input.targets)


    sv = tf.train.Supervisor(logdir=args.model_dir,
                             save_summaries_secs=10,
                             save_model_secs=10)
    with sv.managed_session() as session:
      train_input.start_queue_thread(session)
      for i in range(params['max_max_epoch']):
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     epoch_size=train_batches.epoch_size,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity = run_epoch(session, mvalid,
                                     epoch_size=valid_batches.epoch_size)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      train_input.shutdown(session)

if __name__ == "__main__":
    main()
