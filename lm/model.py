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

""" LSTM language model based on the PTB model from TensorFlow tutorial.

Hyperparameters should go in a separate hparams.yaml file. A known good
configuration is saved in this project directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lm.reader as reader

import lm.opt as opt
import inspect
import time

import numpy as np
import tensorflow as tf
import yaml


logging = tf.logging
ModeKeys = tf.contrib.learn.ModeKeys
DEFAULT_MAX_GRAD_NORM = 5

class InputData(object):
  """The input data."""

  def __init__(self, params, data, name=None):
    self.batch_size = batch_size = params['batch_size']
    self.num_steps = num_steps = params['unroll_length']
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.example_producer(
        data, batch_size, num_steps, name=name)


def add_recall_at_k_summaries(logits, targets, ks=[1, 5, 10]):
    flat_logits = tf.reshape(logits, (-1, logits.shape[-1].value))
    flat_targets = tf.reshape(targets, (-1,))
    for k in ks:
        at_k = tf.to_float(tf.nn.in_top_k(flat_logits, flat_targets,
            k))
        tf.summary.scalar('hit_%d' % k, tf.reduce_mean(at_k))


def clipped_train_op(loss, var_list, params, add_summaries=True):
    max_grad_norm = params.get('max_grad_norm', DEFAULT_MAX_GRAD_NORM)
    optimizer = opt.create_optimizer(params['opt_method'],
                                     params['opt_params'])
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads, tvars = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads,
                                      max_grad_norm)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    if add_summaries:
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradient', grad)

    return train_op


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

    logits = tf.reshape(logits, [batch_size,
                                 num_steps,
                                 vocab_size])
    self._final_state = state
    self.input_tokens = tokens
    self.token_probability = tf.nn.softmax(logits)
    if mode == ModeKeys.INFER:
        return

    # == Below here, |targets| are guaranteed to be meaningful. ==
    add_recall_at_k_summaries(logits, targets, ks=[1, 5, 10])
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
            tf.ones_like(targets, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)
    tf.summary.scalar('loss', loss)

    # Save references to important tensors
    self._cost = loss*num_steps

    if not is_training:
      return

    tvars = tf.trainable_variables()
    self._train_op = clipped_train_op(loss, tvars, params)

    # Not actually using this...
    self._lr = tf.Variable(0.0,
                           trainable=False)
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

