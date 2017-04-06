"""
    Generates a sample using the language model in a given checkpoint.

    Sample usage:

        python generate.py --model_dir /tmp/models --length 200
"""
import argparse
import tensorflow as tf
import numpy as np
from model import model_fn
from sample import sample
from reader import load_vocab
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint",
                        default="/tmp/legislation/model.ckpt")
    parser.add_argument("--data_path",
        default="/home/kevin/projects/legislation-project/uscode/processed-data")
    parser.add_argument("--temp", '-T', type=float, default=1.0)
    parser.add_argument("--hyperparams", default=None,
                        help="yaml file of hyperparameters")
    parser.add_argument("--length", type=int, default=200,
                        help="Max number of tokens to generate.")
    parser.add_argument("--output", '-o', type=str, default=None)

    return parser.parse_args()

class ModelStepper(object):
    def __init__(self, model_fn, session, params, model_checkpoint):
        self.token_input = tf.placeholder(tf.int32, shape=(params['batch_size'],1))
        self.model_fn_ops = model_fn({'tokens': self.token_input }, None,
                                     tf.contrib.learn.ModeKeys.INFER, params)
        self.state = (np.zeros((params['batch_size'], params['embedding_dim']),
                               dtype=np.float32),
                      np.zeros((params['batch_size'], params['embedding_dim']),
                               dtype=np.float32))

        saver = tf.train.Saver()
        saver.restore(session, model_checkpoint)
        print("Restored model from %s", model_checkpoint)
        self.session = session

    def reset_state(self):
        self.state = (np.zeros((params['batch_size'], params['embedding_dim']),
                               dtype=np.float32),
                      np.zeros((params['batch_size'], params['embedding_dim']),
                               dtype=np.float32))

    def run_step(self, input_token):
        """ Runs single step of model and returns probability distribution over
            tokens.
        """
        preds = self.model_fn_ops.predictions
        feed_dict = {}
        feed_dict[self.token_input] = [[input_token]]
        feed_dict['init_c_state:0'] = self.state[0]
        feed_dict['init_h_state:0'] = self.state[1]
        probs, self.state = self.session.run(
                             [preds['probability'], preds['final_state']],
                             feed_dict=feed_dict)

        return probs


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

    sess = tf.Session()
    stepper = ModelStepper(model_fn, sess, params, args.model_checkpoint)
    token_ids = [0]
    for _ in xrange(args.length):
        probs = stepper.run_step(token_ids[-1])
        token_id = sample(probs[:, 0, :], temp=args.temp)[0]
        token_ids.append(token_id)

    if not args.output:
        print ' '.join(vocab.get_token(idx) for idx in token_ids)
    else:
        with open(args.output, 'w') as fp:
            fp.write(' '.join(vocab.get_token(idx) for idx in token_ids))

if __name__ == "__main__":
    main()
