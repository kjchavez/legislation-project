from __future__ import division
from __future__ import print_function

import tensorflow as tf
from os.path import join
from preprocess.vocab import Vocabulary

def _load_ids_from_file(filename):
    with open(filename) as fp:
        return [int(line.strip()) for line in fp]

def _raw_data(datadir):
    train_file = join(datadir, 'train.txt')
    valid_file = join(datadir, 'valid.txt')
    test_file = join(datadir, 'test.txt')
    vocab_file = join(datadir, 'vocabulary.txt')

    return _load_ids_from_file(train_file), _load_ids_from_file(valid_file), \
           _load_ids_from_file(test_file), Vocabulary.fromfile(vocab_file)

def example_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw token ids.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: a list of integer token ids
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "Producer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
