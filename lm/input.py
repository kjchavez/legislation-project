# TensorFlow Input Pipelines for Large Data Sets
# ischlag.github.io
# TensorFlow 0.11, 07.11.2016

import tensorflow as tf
import numpy as np
import threading

class SequentialInputData(object):
    def __init__(self, raw_data_generator, batch_size, num_steps, queue_capacity=100,
            dtype=tf.float32, enqueue_batch_size=10):
        self.raw_data_generator = raw_data_generator
        enqueue_shape = [enqueue_batch_size, batch_size, num_steps]
        queue_input_data = tf.placeholder(dtype, shape=enqueue_shape)
        queue_input_target = tf.placeholder(dtype, shape=enqueue_shape)

        self._queue = tf.FIFOQueue(capacity=queue_capacity, dtypes=[dtype, dtype],
                shapes=[[batch_size, num_steps], [batch_size, num_steps]])
        self.enqueue_op = self._queue.enqueue_many([queue_input_data, queue_input_target])
        self.x, self.y = self._queue.dequeue()

        self._queue_input_data = queue_input_data
        self._queue_input_target = queue_input_target
        self._cancelled = False

    def enqueue(self, sess):
      """ Iterates over our data puts small junks into our queue."""
      while True:
        if self._cancelled:
            break

        print("starting to write into queue")
        try:
          curr_data, curr_target = next(self.raw_data_generator)
        except StopIteration:
            print("Reached end of generator.")
            break

        sess.run(self.enqueue_op, feed_dict={self._queue_input_data: curr_data,
                                        self._queue_input_target: curr_target})
        print("added to the queue")

      print("finished enqueueing")

    def start_enqueue_thread(self, sess):
        enqueue_thread = threading.Thread(target=self.enqueue, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()

    def shutdown(self, sess):
        self._cancelled = True
        sess.run(self._queue.close(cancel_pending_enqueues=True))

def sanity_check():
    def fake_data_gen():
        for i in xrange(100):
            print ("Called generator...")
            x = np.ones((10, 32, 20), dtype=np.int32) * i
            y = np.ones((10, 32, 20), dtype=np.int32) * i
            yield x, y

    sess = tf.Session()
    d = SequentialInputData(fake_data_gen(), 32, 20, dtype=tf.int32)
    d.start_enqueue_thread(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    for i in xrange(100*10):
        run_options = tf.RunOptions(timeout_in_ms=40000)
        curr_data_batch, curr_target_batch = sess.run([d.x, d.y], options=run_options)
        print(curr_data_batch)

    # shutdown everything to avoid zombies
    d.shutdown(sess)
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    sanity_check()
