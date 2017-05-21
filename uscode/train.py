import tensorflow as tf
from tensorflow.python import debug as tf_debug
import reader
from model import LanguageModel
import argparse
import datetime
import os
import shutil
import yaml

tf.logging.set_verbosity(tf.logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/tmp/legislation-model")
    parser.add_argument("--reset", action='store_true',
                        help="If true, clears any data in 'model_dir' before "
                             "training")
    parser.add_argument("--data_path",
        default="/home/kevin/projects/legislation-project/output")
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--debug", action='store_true')

    # Hyperparameters can be set individually via commandline args...
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--unroll_steps", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # ...or with a YAML file containing all the values.
    parser.add_argument("--hyperparams", default=None,
                        help="yaml file of hyperparameters")

    return parser.parse_args()

def save_params(params, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = 'params.{:%Y-%m-%d.%H:%M:%S}.yaml'.format(datetime.datetime.now())
    with open(os.path.join(model_dir, filename), 'w') as fp:
        yaml.dump(params, fp, default_flow_style=False)

args = parse_args()

if args.reset and os.path.exists(args.model_dir):
	shutil.rmtree(args.model_dir)

if args.hyperparams:
    with open(args.hyperparams) as fp:
        params = yaml.load(fp)
else:
    params = {}
    params['embedding_dim'] = args.embedding_dim
    params['batch_size'] = args.batch_size
    params['unroll_length'] = args.unroll_steps
    params['learning_rate'] = args.learning_rate

required = ('embedding_dim', 'batch_size', 'unroll_length', 'opt_method',
        'opt_params', 'num_layers')
if not all(param in params for param in required):
    raise ValueError("Required parameters are missing.")
    sys.exit(1)

train, valid, test, vocab = reader._raw_data(args.data_path)

# Actual vocab size is determined after loading data 
params['vocab_size'] = vocab.size()

# Save params to model directory for future reference.
save_params(params, args.model_dir)

config = tf.contrib.learn.RunConfig(
            log_device_placement=False,
            keep_checkpoint_every_n_hours=1,
            save_checkpoints_secs=60,
            # save_checkpoints_steps=1000,
            keep_checkpoint_max=5,
            gpu_memory_fraction=0.7,
            tf_random_seed=0,
        )


model = LanguageModel(params)
estimator = tf.contrib.learn.Estimator(
                model_fn=model.model_fn,
                model_dir=args.model_dir,
                config=config,
                params=params)

def train_input_fn():
    return reader.example_producer(train, params['batch_size'],
                                   params['unroll_length'])


hooks = []
if args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    hooks.append(debug_hook)

estimator.fit(input_fn=train_input_fn,
              steps=args.train_steps,
              monitors=hooks)
