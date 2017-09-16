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

def _model_instance_from_dir(model_dir):
    """ Returns model and instance name for the given |model_dir|. """
    return os.path.basename(model_dir.rstrip('/')).split('-')

def _model_dir(model_name, instance_name, path=None):
    """ Creates a directory for a (model, instance) if it doesn't exist.
        Returns absolute path for directory.
    """
    dirname = "%s-%s" % (model_name, instance_name)
    if path:
        dirname = os.path.join(path, dirname)

    if not os.path.exists(dirname):
        logging.info("Creating model directory: %s", model_dir)
        os.makedirs(dirname)

    return os.path.abspath(dirname)


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


def save_eval_results(model_dir, valid_filepattern, config=None,
                      outputdir=None):
    if not os.path.exists(model_dir):
        logging.error("Model directory does not exist: %s", model_dir)
        return

    hparams_file = join(model_dir, SAVED_HPARAMS_FILENAME)
    if not os.path.exists(hparams_file):
        logging.error("%s not found", SAVED_HPARAMS_FILENAME)
        return

    with open(hparams_file) as fp:
        hparams = yaml.load(fp)

    # To add special evaluation hooks
    hparams['save_eval'] = True

    # If dev set filepattern not specified use the one bundled with the model checkpoints.
    if not valid_filepattern:
        valid_filepattern = join(hparams['datadir'], 'valid.tfrecord')
        logging.info("Using default filepattern: %s", valid_filepattern)

    valid_input_fn = lambda: input_pipeline.inputs(valid_filepattern, hparams['batch_size'],
                                                   num_epochs=1)
    model_name, _ = _model_instance_from_dir(model_dir)
    model_fn = get_model_fn_by_name(model_name)

    if not outputdir:
        outputdir = join(model_dir, "error_analysis")
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
                                       config=None, params=hparams)
    eval_hooks = []
    metrics = estimator.evaluate(valid_input_fn, hooks=eval_hooks)
    print(metrics)

def get_hparams(model_dir):
    """ Loads YAML file from |model_dir| if it exists. """
    saved_hparams = join(model_dir, SAVED_HPARAMS_FILENAME)
    if os.path.exists(model_dir):
        # Hyperparameters should be saved, so just restore them.
        with open(saved_hparams) as fp:
            return yaml.load(fp)

def add_training_process_args(parser):
    parser.add_argument('--eval_every_n', type=int, default=100000,
                        help='number of train steps between evals')
    parser.add_argument('--export_savedmodel', type=bool, default=True,
                        help="export of savedmodel at every eval")


def parse_args():
    parser = argparse.ArgumentParser("Commandline tool for training and evaluating models for " \
                                     "vote prediction")
    subparsers = parser.add_subparsers()

    # Train new model.
    new_model_parser = subparsers.add_parser('new_model', help="Start training for new model")
    new_model_parser.add_argument('--model', '-m', required=True,
                                  help="which model to use")
    new_model_parser.add_argument('--instance_name', '-i', required=True,
                                  help="name for this instance of the model")
    new_model_parser.add_argument('--datadir', '-d', default="data",
                        help="directory holding train/dev/test data. " \
                             "Assumed to have train.tfrecord and dev.tfrecord.")
    new_model_parser.add_argument('--hparams', default=None, required=True,
                                  help='yaml file of hyperparameters')
    add_training_process_args(new_model_parser)
    new_model_parser.set_defaults(func=start_new_model)

    # Continue training.
    ct_parser = subparsers.add_parser('continue', help="Continue training model")
    ct_parser.add_argument("--model_dir", required=True,
                           help="Directory of model checkpoints.")
    add_training_process_args(ct_parser)
    ct_parser.set_defaults(func=continue_training)

    # Eval.
    eval_parser = subparsers.add_parser('eval', help="evaluate a model")
    eval_parser.add_argument("--model_dir", required=True,
                             help="Directory of model checkpoints.")
    eval_parser.add_argument("--data", default=None,
                             help="TFRecord holding dev set examples")
    eval_parser.add_argument("--outputdir", default=None,
                             help="Directory to save results. Defaults to " \
                                  "%model_dir%/error_analysis.")
    eval_parser.add_argument("--verbose", '-v', action='store_true',
                             help="stores all examples and predictions on dev" \
                                  " set for error analysis.")
    eval_parser.set_defaults(func=evaluate_model)
    return parser.parse_args()

def evaluate_model(args):
    config = tf.estimator.RunConfig()
    save_eval_results(args.model_dir, args.data, config=config, outputdir=args.outputdir)

def continue_training(args):
    if not os.path.exists(args.model_dir):
        logging.error("Directory does not exist: %s", args.model_dir)
        return

    config = tf.estimator.RunConfig()
    hparams = get_hparams(args.model_dir)
    datadir = hparams['datadir']
    model, instance_name = _model_instance_from_dir(args.model_dir)
    train(model, instance_name, join(datadir, 'train.tfrecord'),
          join(datadir, 'valid.tfrecord'),
          hparams=hparams, config=config, eval_every_n=args.eval_every_n,
          export=args.export_savedmodel)

def start_new_model(args):
    config = tf.estimator.RunConfig()
    model_dir = join("results", "%s-%s" % (args.model, args.instance_name))
    if os.path.exists(model_dir):
        logging.info("Force reset. Removing directory: %s", model_dir)
        shutil.rmtree(model_dir)

    # Get the hyperparameters.
    # TODO(kjchavez): There may be a TensorFlow idiomatic way to save and restore hyperparameters
    # from model checkpoints. Investigate.
    with open(args.hparams) as fp:
        hparams = yaml.load(fp)

    # Somewhat hacky overrides.
    vocab_filename = join(args.datadir, 'vocab.txt')
    hparams['vocab_filename'] = os.path.abspath(vocab_filename)
    hparams['datadir'] = os.path.abspath(args.datadir)
    hparams['save_eval'] = False

    train(args.model, args.instance_name, join(args.datadir, 'train.tfrecord'),
          join(args.datadir, 'valid.tfrecord'),
          hparams=hparams, config=config, eval_every_n=args.eval_every_n,
          export=args.export_savedmodel)

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
