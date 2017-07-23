import tensorflow as tf

def feature_spec():
    """ Example input.

    {'VoterParty': 'republican', 'BillTitle': u'Making appropriations for the Departments of Labor,
    Health and Human Services, and Education, and related agencies, for the fiscal year ending September
    30, 1997, and for other purposes.', 'VoterState': u'CO', 'SponsorParty': 'republican', 'BillId':
        u'hr3755-104', 'VoterChamber': u'sen', 'VoterAge': 73, 'Decision': 'Aye'}
    """
    return {
      'SponsorParty': tf.FixedLenFeature([], tf.string),
      'VoterParty': tf.FixedLenFeature([], tf.string),
      'VoterState': tf.FixedLenFeature([], tf.string),
      'VoterAge': tf.FixedLenFeature([], tf.int64),
      'Decision': tf.FixedLenFeature([], tf.string)
    }

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features=feature_spec())

  # Be careful with consistency here..
  features['Decision'] = tf.to_int32(tf.equal(features.pop('Decision'), 'Aye'))
  return features

def inputs(filepattern, batch_size, num_epochs=None):
  """Reads input data num_epochs times.
  Args:
    filepattern: Files to draw inputs from.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(filepattern), num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    features_dict = read_and_decode(filename_queue)

    # We run this in two threads to avoid being a bottleneck.
    features = tf.train.shuffle_batch(
        features_dict, batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    labels = features.pop('Decision')

    return features, labels


if __name__ == "__main__":
    # Run sanity check.
    X, y = inputs("data/train.tfrecord", 2, num_epochs=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        with coord.stop_on_exception():
            while not coord.should_stop():
                x_val, y_val = sess.run([X, y])
                print(x_val, y_val)

            coord.request_stop()
            coord.join(threads)
