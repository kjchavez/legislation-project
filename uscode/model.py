import tensorflow as tf
import reader

data_path = "/home/kevin/projects/legislation-project/uscode/processed-data"

BATCH_SIZE = 4
UNROLL_LEN = 10

train, valid, test, vocab = reader._raw_data(data_path)
x, y = reader.example_producer(train, BATCH_SIZE, UNROLL_LEN)

def model_fn(features, targets, mode, params):
    cell = tf.contrib.rnn.BasicLSTMCell(
            params['embedding_dim'], forget_bias=0.0, state_is_tuple=True)

    initial_state = cell.zero_state(params['batch_size'], tf.float32)
    embedding = tf.get_variable("embedding", [params['vocab_size'], params['embedding_dim']],
                                dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, features)
    print "Inputs shape:", inputs.get_shape()
    inputs = tf.unstack(inputs, num=params['num_steps'], axis=1)
	# tf.contrib.rnn.static_rnn

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
    logits = tf.reshape(logits, [params['batch_size'], params['num_steps'],
                        params['vocab_size']])
    print logits.get_shape()

    loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
            tf.ones_like(targets, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

    predictions = tf.arg_max(logits, 2)
    lr = params['learning_rate']
    max_grad_norm = 5  # params['max_grad_norm']
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    # Return predictions/loss/train_op/eval_metric_ops
	# ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops)
    return tf.contrib.learn.ModelFnOps(mode, predictions, loss, train_op, {})

z = model_fn(x, y, None, {'vocab_size' : vocab.size(), 'embedding_dim': 8,
                          'batch_size': BATCH_SIZE, 'num_steps': UNROLL_LEN,
                          'learning_rate': 0.0001})
init_op = tf.global_variables_initializer()

coord = tf.train.Coordinator()
with tf.Session() as sess:
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    _x, _y, _z = sess.run([x, y, z.loss])

coord.request_stop()
coord.join(threads)

print "X"
print _x
print "Y"
print _y
print "Z"
print _z
