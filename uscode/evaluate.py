"""
    Evaluates a checkpoint stored in the model directory.

    Note: You should generally pass the same file of hyperparameters as used
    during training.

"""
import tensorflow as tf
import argparse
import yaml
import reader
from model import model_fn

tf.logging.set_verbosity(tf.logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/tmp/legislation-model")
    parser.add_argument("--data_path",
        default="/home/kevin/projects/legislation-project/output")
    parser.add_argument("--checkpoint", '-c', default=None)
    parser.add_argument("--steps", '-s', type=int, default=None)

    parser.add_argument("--hyperparams", default=None,
                        help="yaml file of hyperparameters")

    return parser.parse_args()

def _print_results(results_dict):
    print "=============== RESULTS ================="
    for key, value in results_dict.items():
        print key.ljust(20), value


def main():
    args = parse_args()
    train, valid, test, vocab = reader._raw_data(args.data_path)

    if args.hyperparams:
        with open(args.hyperparams) as fp:
            params = yaml.load(fp)

    params['vocab_size'] = vocab.size()

    estimator = tf.contrib.learn.Estimator(
                    model_fn=model_fn,
                    model_dir=args.model_dir,
                    params=params)

    def valid_input_fn():
        return reader.example_producer(valid, params['batch_size'],
                                       params['unroll_length'])

    if not args.steps:
        args.steps = len(valid) / params['batch_size']

    # Note the results of evaluation are saved to <model_dir>/eval
    results = estimator.evaluate(input_fn=valid_input_fn,
                                 steps=args.steps,
                                 checkpoint_path=args.checkpoint)

    _print_results(results)


if __name__ == "__main__":
    main()
