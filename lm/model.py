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
from lm.input import QueuedInputData

import lm.opt as opt
import inspect
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import yaml


logging = tf.logging
ModeKeys = tf.contrib.learn.ModeKeys
DEFAULT_MAX_GRAD_NORM = 5
DEFAULT_MAX_SAMPLE_LENGTH = 1000

# TODO(kjchavez): Bundle these with the model or at least in a separate
# constants file.
OOV_ID = 0
SOB_TOKEN_ID = 1
EOB_TOKEN_ID = 2

# TODO(kjchavez): This doesn't belong in this file. Also may be redundant now
# that we have other input classes.
class InputData(object):
  """The input data."""

  def __init__(self, params, batched_dataset, name=None):
    self.batch_size = batch_size = params['batch_size']
    self.num_steps = num_steps = params['unroll_length']

    # We're only adding a single batch to the queue at a time, but first
    # dimension of the 'enqueue batch' should reflect that.
    def expand_dims():
        for x, y in batched_dataset.generator():
            yield (np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))

    qid = QueuedInputData(expand_dims(),
                          data_shape=(batch_size, num_steps),
                          target_shape=(batch_size, num_steps),
                          dtype=tf.int32,
                          enqueue_batch_size=1)
    self.input_data = {'tokens': qid.x}
    self.targets = qid.y
    self.epoch_size = batched_dataset.epoch_size
    self.qid = qid

  def start_queue_thread(self, sess):
      self.qid.start_enqueue_thread(sess)

  def shutdown(self, sess):
      self.qid.shutdown(sess)


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
        if 'learning_rate' in params['opt_params']:
            tf.summary.scalar('learning_rate',
                              params['opt_params']['learning_rate'])

        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradient', grad)

    return train_op

class SamplingEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  def __init__(self, embedding, start_tokens, end_token, temperature=1.0,
          params={}):
    super(SamplingEmbeddingHelper, self).__init__(embedding, start_tokens,
              end_token)
    self.temperature = temperature
    self.seed = params.get("seed", None)

  def sample(self, time, outputs, state, name=None):
    """Override sample method to get variety"""
    del time, state  # unused by sample_fn
    # Outputs are logits, use multinomial to sample from distribution. 
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    # TODO(kjchavez): Consider what to do if we sample the out-of-vocab token.
    sample_ids = tf.reshape(tf.to_int32(tf.multinomial(outputs, 1,
                                                       seed=self.seed)), [-1])
    return sample_ids


class LanguageModel(object):
  """The Language model."""

  def __init__(self, features, targets, mode, params, dtype=tf.float32):
    tokens = features['tokens']

    self._params = params
    batch_size = params['batch_size']
    num_steps = params['unroll_length']
    size = params['embedding_dim']
    vocab_size = params['vocab_size']
    keep_prob = params['keep_prob']
    num_layers = params['num_layers']
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    global_step = tf.contrib.framework.get_or_create_global_step()

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

    mapping_string = tf.constant(params['vocab'])
    self.table = tf.contrib.lookup.index_to_string_table_from_tensor(
                mapping_string, default_value="UNKNOWN")
    if mode == "GENERATE":
        # We need to feed an input at each step from the output of the previous
        # step.
        self.temperature = tf.placeholder(tf.float32)
        helper = SamplingEmbeddingHelper(
          embedding=embedding,
          start_tokens=tf.tile([SOB_TOKEN_ID], [batch_size]),
          end_token=EOB_TOKEN_ID,
          temperature=self.temperature)
    else:
        # NOTE(kjchavez): By default, the inputs to TrainingHelper are assumed
        # to be batch major. Use time_major=True if you care to flip it.
        helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=tf.tile([num_steps], [batch_size]))

    # Create softmax layer.
    output_layer = Dense(vocab_size)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=cell.zero_state(batch_size, dtype),
        output_layer=output_layer)
    outputs, state = tf.contrib.seq2seq.dynamic_decode(
       decoder=decoder,
       output_time_major=False,
       impute_finished=True,
       maximum_iterations=params.get('max_sample_length',
           DEFAULT_MAX_SAMPLE_LENGTH))

    self.output_token_ids = outputs.sample_id
    self.output_tokens = self.table.lookup(tf.to_int64(outputs.sample_id))
    logits = outputs.rnn_output

    self._final_state = state
    self.input_tokens = tokens
    self.token_probability = tf.nn.softmax(logits)
    if mode in (ModeKeys.INFER, "GENERATE"):
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
  def train_op(self):
    return self._train_op


def run_epoch(session, model, epoch_size, eval_op=None, verbose=False):
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

  for step in range(epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 100) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

