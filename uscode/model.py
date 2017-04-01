import tensorflow as tf
import reader
import math

def model_fn(features, targets, mode, params):
    cell = tf.contrib.rnn.BasicLSTMCell(
            params['embedding_dim'], forget_bias=0.0, state_is_tuple=True)

    initial_state = cell.zero_state(params['batch_size'], tf.float32)
    embedding = tf.get_variable("embedding", [params['vocab_size'], params['embedding_dim']],
                                dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, features)
    inputs = tf.unstack(inputs, num=params['unroll_length'], axis=1)

    # Note: This always feeds the 'true' next item in the sequence as input
    # into the RNN. We may want to have a mode where we feed the predicted
    # token as input.
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                               initial_state=initial_state)
    output = tf.stack(outputs)
    output = tf.reshape(output, [-1, params['embedding_dim']])
    softmax_w = tf.get_variable(
        "softmax_w", [params['embedding_dim'], params['vocab_size']], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [params['vocab_size']], dtype=tf.float32)
    logits = tf.matmul(output, softmax_w) + softmax_b
    logits = tf.reshape(logits, [params['batch_size'], params['unroll_length'],
                        params['vocab_size']])

    predictions = tf.arg_max(logits, 2)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                tf.ones_like(targets, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
        lr = params['learning_rate']
        max_grad_norm = 5  # params['max_grad_norm']
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

    else:
        loss = None
        train_op = None

    # Return predictions/loss/train_op/eval_metric_ops
    return tf.contrib.learn.ModelFnOps(mode, predictions, loss, train_op, {})


def sanity_check():
    print("============== RUNNING SANITY CHECK =====================")
    data_path = "/home/kevin/projects/legislation-project/uscode/processed-data"

    BATCH_SIZE = 4
    UNROLL_LEN = 10

    train, valid, test, vocab = reader._raw_data(data_path)

    params = {}
    params['vocab_size'] = vocab.size()
    params['embedding_dim'] = 8
    params['batch_size'] = BATCH_SIZE
    params['unroll_length'] = UNROLL_LEN
    params['learning_rate'] = 0.0001

    x, y = reader.example_producer(train, BATCH_SIZE, UNROLL_LEN)
    z = model_fn(x, y, tf.contrib.learn.ModeKeys.TRAIN,
                 {'vocab_size' : vocab.size(), 'embedding_dim': 8,
                  'batch_size': BATCH_SIZE, 'unroll_length': UNROLL_LEN,
                  'learning_rate': 0.0001})
    init_op = tf.global_variables_initializer()

    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init_op)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        _x, _y, _z = sess.run([x, y, z.loss])

    coord.request_stop()
    coord.join(threads)

    print "Average loss for rand init model: ", _z
    print "Should be around: ", math.log(params['vocab_size'])

if __name__ == "__main__":
    sanity_check()
