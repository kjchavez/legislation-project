import tensorflow as tf
import argparse
import logging
import os
from os.path import join
import shutil
import yaml

import models
import input_pipeline

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

SAVED_HPARAMS_FILENAME = "hparams.yaml"


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

    hparams_file = join(model_dir, SAVED_HPARAMS_FILENAME)
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

    hparams_file = join(model_dir, SAVED_HPARAMS_FILENAME)
    if not os.path.exists(hparams_file):
        logging.error("%s not found", SAVED_HPARAMS_FILENAME)
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

def get_hparams(model_dir, hparams_file):
    """ Loads YAML file from |model_dir| if it exists, else from |hparams_file|.
    """
    hparams = {}
    saved_hparams = join(model_dir, SAVED_HPARAMS_FILENAME)
    if os.path.exists(model_dir):
        # Hyperparameters should be saved, so just restore them.
        with open(saved_hparams) as fp:
            hparams = yaml.load(fp)
    else:
        # Accept hparams from commandline arg.
        if hparams_file:
            with open(hparams_file) as fp:
                hparams = yaml.load(fp)
        else:
            logging.warning("No hparams specified, is that intentional?")

    return hparams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True,
                        help="which model to use")
    parser.add_argument('--datadir', '-d', default="data",
                        help="directory holding train/dev/test data. " \
                             "Assumed to have train.tfrecord and dev.tfrecord.")
    parser.add_argument('--instance_name', '-i', required=True,
                        help="name for this instance of the model")
    parser.add_argument('--reset', '-r', action='store_true', default=False)
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

    model_dir = join("results", "%s-%s" % (args.model, args.instance_name))
    if args.reset and os.path.exists(model_dir):
        logging.info("Force reset. Removing directory: %s", model_dir)
        shutil.rmtree(model_dir)

    # Get the hyperparameters.
    # TODO(kjchavez): There may be a TensorFlow idiomatic way to save and restore hyperparameters
    # from model checkpoints. Investigate.
    hparams = get_hparams(model_dir, args.hparams)

    # Somewhat hacky overrides.
    vocab_filename = join(args.datadir, 'vocab.txt')
    hparams['vocab_filename'] = vocab_filename
    hparams['save_eval'] = args.save_eval

    train(args.model, args.instance_name, join(args.datadir, 'train.tfrecord'),
          join(args.datadir, 'valid.tfrecord'),
          hparams=hparams, config=config, eval_every_n=args.eval_every_n,
          export=args.export_savedmodel)

if __name__ == "__main__":
    main()
