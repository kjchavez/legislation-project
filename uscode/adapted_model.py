# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a Language LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- unroll_length - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
Language dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --model_dir=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uscode.reader as reader

import inspect
import time

import numpy as np
import tensorflow as tf
import yaml


logging = tf.logging
ModeKeys = tf.contrib.learn.ModeKeys

class InputData(object):
  """The input data."""

  def __init__(self, params, data, name=None):
    self.batch_size = batch_size = params['batch_size']
    self.num_steps = num_steps = params['unroll_length']
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.example_producer(
        data, batch_size, num_steps, name=name)


class LanguageModel(object):
  """The Language model."""

  def __init__(self, features, targets, mode, params, epoch_size=None, dtype=tf.float32):
    tokens = features['tokens']

    self._params = params
    self.epoch_size = epoch_size
    batch_size = params['batch_size']
    num_steps = params['unroll_length']
    size = params['embedding_dim']
    vocab_size = params['vocab_size']
    keep_prob = params['keep_prob']
    num_layers = params['num_layers']
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True,
          reuse=tf.get_variable_scope().reuse)

    attn_cell = lstm_cell
    if is_training and keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, dtype)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=dtype)
      inputs = tf.nn.embedding_lookup(embedding, tokens)

    if is_training and keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)

    inputs = tf.unstack(inputs, num=num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs, initial_state=self._initial_state)

    # Note that here we 'stack' it in batch major order, so we won't
    # need to transpose this later.
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])

    # Create softmax layer.
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=dtype)
    tf.summary.histogram('softmax_b', softmax_b)
    tf.summary.histogram('softmax_w', softmax_w)

    logits = tf.matmul(output, softmax_w) + softmax_b

    # From model.py
    logits = tf.reshape(logits, [batch_size,
                                 num_steps,
                                 vocab_size])
    self._final_state = state
    self.input_tokens = tokens
    self.token_probability = tf.nn.softmax(logits)
    if mode == ModeKeys.INFER:
        return

    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
            tf.ones_like(targets, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)
    tf.summary.scalar('loss', loss)

    # Save references to important tensors
    self._cost = loss*num_steps

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      params['max_grad_norm'])
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def batch_size(self):
      return self._params['batch_size']

  @property
  def num_steps(self):
      return self._params['unroll_length']

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.num_steps

    if verbose and step % (model.epoch_size // 100) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

