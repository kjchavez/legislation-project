import tensorflow as tf
import argparse
import os

import models
import input_pipeline

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn_by_name(name):
    submodule = getattr(models, name)
    return submodule.model_fn


def train(model_name, instance_name, train_filepattern, valid_filepattern, hparams, config=None,
          eval_every_n=10000):
    train_input_fn = lambda: input_pipeline.inputs(train_filepattern, hparams['batch_size'],
                                                   num_epochs=None)
    valid_input_fn = lambda: input_pipeline.inputs(valid_filepattern, hparams['batch_size'],
                                                   num_epochs=1)

    model_fn = get_model_fn_by_name(model_name)

    # TODO(kjchavez): If the instance_name for the model already exists, hyperparams must match or
    # be loaded.
    model_dir = os.path.join('results', '%s-%s' % (model_name, instance_name))
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
                                       config=config, params=hparams)

    while True:
        estimator.train(train_input_fn, max_steps=eval_every_n)
        metrics = estimator.evaluate(valid_input_fn)
        print(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True,
                        help="which model to use")
    parser.add_argument('--instance_name', '-i', required=True,
                        help="name for this instance of the model")
    parser.add_argument('--eval_every_n', type=int, default=100000,
                        help='number of train steps between evals')
    return parser.parse_args()

args = parse_args()
config = tf.estimator.RunConfig()
train(args.model, args.instance_name, 'data/votes.tfrecord', 'data/votes.tfrecord',
      hparams={'batch_size': 100}, config=config, eval_every_n=args.eval_every_n)
