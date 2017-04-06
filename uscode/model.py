import tensorflow as tf
import reader
import math

ModeKeys = tf.contrib.learn.ModeKeys

def model_fn(features, targets, mode, params):
    tokens = features['tokens']

    cell = tf.contrib.rnn.BasicLSTMCell(
            params['embedding_dim'], forget_bias=0.0, state_is_tuple=True)

    zero_state = cell.zero_state(params['batch_size'], tf.float32)

    initial_c_state = tf.placeholder_with_default(zero_state.c,
                                                  zero_state.c.get_shape(),
                                                  name="init_c_state");
    initial_h_state = tf.placeholder_with_default(zero_state.h,
                                                  zero_state.h.get_shape(),
                                                  name="init_h_state");
    initial_state = tf.contrib.rnn.LSTMStateTuple(initial_c_state,
                                                  initial_h_state)
    embedding = tf.get_variable("embedding", [params['vocab_size'], params['embedding_dim']],
                                dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, tokens)
    inputs = tf.unstack(inputs, num=params['unroll_length'], axis=1)

    # Note: This always feeds the 'true' next item in the sequence as input
    # into the RNN. We may want to have a mode where we feed the predicted
    # token as input.
    #
    # To use this model to generate text, we can do one of a few things:
    # 1. Set the unroll length to 1, and only generate a single 'token' at a
    #    time. We will have to add an 'initial_token' placeholder for this. And
    #    it will be slow...
    #    http://stackoverflow.com/questions/36609920/tensorflow-using-lstms-for-generating-text 
    #    http://deeplearningathome.com/2016/10/Text-generation-using-deep-recurrent-neural-networks.html 
    #
    # 2. Use raw_rnn instead of static_rnn here -- with a more customizable
    #    loop fn that can be different at generation time.
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

    predictions = {}
    predictions['tokens'] = tf.arg_max(logits, 2)
    predictions['probability'] = tf.nn.softmax(logits)
    predictions['final_state'] = state
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                tf.ones_like(targets, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)

    if mode == ModeKeys.TRAIN:
        lr = params['learning_rate']
        max_grad_norm = 5  # params['max_grad_norm']
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

    # Add summaries of helpful values.
    if mode == ModeKeys.TRAIN:
        tf.summary.scalar('loss', loss)
        for var in tvars:
            tf.summary.histogram(var.name, var)

        grads = list(zip(grads, tvars))
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)


    # Return predictions/loss/train_op/eval_metric_ops
    return tf.contrib.learn.ModelFnOps(mode, predictions, loss, train_op,
                                       eval_metric_ops)



def sample(distribution, temp=1.0):
    """ Samples from a distribution with a given temperature.

    Args:
        distribution: an M x N numpy array representing M distributions over N
                      possible values.
    """
    M, N = distribution.shape
    coef = 1.0 / temp
    dist = np.power(distribution, coef)
    dist /= np.sum(dist, axis=1, keepdims=True)
    values = np.empty(M)
    for i in xrange(M):
        values[i] = np.random.choice(xrange(N), p=dist[i])

    return values

def sanity_check():
    import numpy as np
    from sample import sample
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
        feed_dict = {}
        feed_dict['init_c_state:0'] = np.random.randn(4, 8)
        feed_dict['init_h_state:0'] = np.random.randn(4, 8)
        _x, _y, _z, probs = sess.run([x, y, z.loss, \
                                      z.predictions['probability']],
                                      feed_dict=feed_dict)

    coord.request_stop()
    coord.join(threads)

    print "Average loss for rand init model: ", _z
    print "Should be around: ", math.log(params['vocab_size'])
    print "Probability distribution:", probs
    print "Shape of probabilities:", probs.shape
    print "A single sample (low T): ", sample(probs[:, 0, :], temp=0.1)
    print "Argmax                 : ", np.argmax(probs[:, 0, :], axis=1)
    print "A single sample (T=1)  : ", sample(probs[:, 0, :], temp=1.0)

if __name__ == "__main__":
    sanity_check()
