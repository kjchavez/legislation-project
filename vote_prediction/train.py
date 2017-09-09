import tensorflow as tf
import argparse
import logging
import os
from os.path import join
import yaml

import models
import input_pipeline

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)


def get_model_fn_by_name(name):
    submodule = getattr(models, name)
    return submodule.model_fn


def train(model_name, instance_name, train_filepattern, valid_filepattern, hparams, config=None,
          eval_every_n=10000, export=True, collect_errors=False):
    train_input_fn = lambda: input_pipeline.inputs(train_filepattern, hparams['batch_size'],
                                                   num_epochs=None)
    valid_input_fn = lambda: input_pipeline.inputs(valid_filepattern, hparams['batch_size'],
                                                   num_epochs=1)

    model_fn = get_model_fn_by_name(model_name)

    # TODO(kjchavez): If the instance_name for the model already exists, hyperparams must match or
    # be loaded.
    model_dir = join('results', '%s-%s' % (model_name, instance_name))
    if not os.path.exists(model_dir):
        logging.info("Creating model directory: %s", model_dir)
        os.makedirs(model_dir)

    hparams_file = join(model_dir, "hparams.yaml")
    if os.path.exists(hparams_file):
        if hparams is not None:
            logging.warning("Overriding hyperparameters with those from checkpoint.")

        with open(hparams_file) as fp:
            hparams = yaml.load(fp)

    with open(hparams_file, 'w') as fp:
        yaml.dump(hparams, fp)
        logging.info("Saved hyperparameters to %s", hparams_file)

    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
                                       config=config, params=hparams)

    exportdir = join('export', '%s-%s' % (model_name, instance_name))
    features = valid_input_fn()[0]
    # But! Let's replace BillTitle with a string!
    features["BillTitle"] = tf.placeholder(tf.string, shape=(1,))
    input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
    while True:
        estimator.train(train_input_fn, steps=eval_every_n)
        metrics = estimator.evaluate(valid_input_fn, hooks=None)
        print(metrics)
        if export:
            estimator.export_savedmodel(exportdir, input_receiver_fn)


def save_eval_results(model_name, instance_name, valid_filepattern, config=None,
                      outputdir=None):
    valid_input_fn = lambda: input_pipeline.inputs(valid_filepattern, hparams['batch_size'],
                                                   num_epochs=1)
    model_fn = get_model_fn_by_name(model_name)
    model_dir = join('results', '%s-%s' % (model_name, instance_name))
    if not os.path.exists(model_dir):
        logging.info("Creating model directory: %s", model_dir)
        os.makedirs(model_dir)

    hparams_file = join(model_dir, "hparams.yaml")
    if not os.path.exists(hparams_file):
        logging.error("hparams.yaml not found")
        return

    with open(hparams_file) as fp:
        hparams = yaml.load(fp)

    # To add special evaluation hooks
    hparams['save_eval'] = True

    if not outputdir:
        outputdir = join(model_dir, "error_analysis")
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
                                       config=None, params=hparams)
    eval_hooks = []
    metrics = estimator.evaluate(valid_input_fn, hooks=eval_hooks)
    print(metrics)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True,
                        help="which model to use")
    parser.add_argument('--datadir', '-d', default="data",
                        help="directory holding train/dev/test data")
    parser.add_argument('--instance_name', '-i', required=True,
                        help="name for this instance of the model")
    parser.add_argument('--eval_every_n', type=int, default=100000,
                        help='number of train steps between evals')
    parser.add_argument('--hparams', default=None,
                        help='yaml file of hyperparameters')
    parser.add_argument('--export_savedmodel', type=bool, default=True,
                        help="disable export of savedmodel at every eval")
    parser.add_argument("--save_eval", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    config = tf.estimator.RunConfig()

    if args.save_eval:
        # Going down a completely different path. Should use 'sub commands' in argparse
        save_eval_results(args.model, args.instance_name,
                          join(args.datadir, 'valid.tfrecord'), config=config)
        return


    if args.hparams is None:
        logging.info("No hparams specified, will attempt to load from checkpoint.")

    with open(args.hparams) as fp:
        hparams = yaml.load(fp)

    vocab_filename = join(args.datadir, 'vocab.txt')
    hparams['vocab_filename'] = vocab_filename
    hparams['save_eval'] = args.save_eval

    train(args.model, args.instance_name, join(args.datadir, 'train.tfrecord'),
          join(args.datadir, 'valid.tfrecord'),
          hparams=hparams, config=config, eval_every_n=args.eval_every_n,
          export=args.export_savedmodel)

if __name__ == "__main__":
    main()
