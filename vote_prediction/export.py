import tensorflow as tf
import argparse
import os
import yaml

import models
import input_pipeline

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn_by_name(name):
    submodule = getattr(models, name)
    return submodule.model_fn


def get_model_dir(model_name, instance_name):
    return os.path.join('results', '%s-%s' % (model_name, instance_name))


def export(model_name, instance_name, hparams, exportdir='export'):
    model_dir = os.path.join('results', '%s-%s' % (model_name, instance_name))
    model_fn = get_model_fn_by_name(model_name)
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir,
                                       params=hparams)
    # TODO(kjchavez): Replace this with something generic, based on the feature spec from
    # input_pipeline.py
    fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {"sponsor_party": tf.placeholder(tf.string, shape=(1,)),
         "member_party": tf.placeholder(tf.string, shape=(1,)) },
            default_batch_size=None
    )

    # NOTE: This will fail unless we provide 'export_outputs' in the EstimatorSpec
    estimator.export_savedmodel(exportdir, fn)


def load_hparams(model_name, instance_name):
    model_dir = get_model_dir(model_name, instance_name)
    with open(os.path.join(model_dir, 'hparams.yaml')) as fp:
        return yaml.load(fp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True,
                        help="which model to use")
    parser.add_argument('--instance_name', '-i', required=True,
                        help="name for this instance of the model")
    return parser.parse_args()

args = parse_args()
hparams = load_hparams(args.model, args.instance_name)
export(args.model, args.instance_name, hparams=hparams)