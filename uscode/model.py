import tensorflow as tf
import reader
import math
import opt

ModeKeys = tf.contrib.learn.ModeKeys
DEFAULT_MAX_GRAD_NORM = 5

def clipped_train_op(loss, var_list, params):
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
    return train_op, grads

class LanguageModel(object):
    def __init__(self, params):
        self.cell = None
        self.initial_state = None

    def model_fn(self, features, targets, mode, params):
        tokens = features['tokens']

        embedding = tf.get_variable("embedding", [params['vocab_size'], params['embedding_dim']],
                                    dtype=tf.float32)

        inputs = tf.nn.embedding_lookup(embedding, tokens)
        inputs = tf.unstack(inputs, num=params['unroll_length'], axis=1)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                    params['embedding_dim'], state_is_tuple=True)

        # Necessary?
        init_scale = 0.05
        initializer = tf.random_uniform_initializer(-init_scale,
                                                    init_scale)

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in
                xrange(params['num_layers'])],
                    state_is_tuple=True)


        self.cell = cell
        self.initial_state = cell.zero_state(params['batch_size'], tf.float32)
        # Note: This always feeds the 'true' next item in the sequence as input
        # into the RNN. We may want to have a mode where we feed the predicted
        # token as input.
        # Also, do we want to build a multi-layer RNN?
        outputs, state = tf.contrib.rnn.static_rnn(self.cell, inputs,
                                   initial_state=self.initial_state)

        output = tf.stack(outputs)
        # Do I need to reorder? This is now TIME major? i.e unroll_length?
        print "Output", output
        output = tf.reshape(output, [-1, params['embedding_dim']])
        softmax_w = tf.get_variable(
            "softmax_w", [params['embedding_dim'], params['vocab_size']],
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                mode='FAN_OUT'),
            dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [params['vocab_size']],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(output, softmax_w) + softmax_b

        # Is this order correct?
        logits = tf.reshape(logits, [params['unroll_length'],
                            params['batch_size'],
                            params['vocab_size']])
        logits = tf.transpose(logits, [1, 0, 2])

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
            loss += params['l2_reg'] * tf.nn.l2_loss(softmax_w)

        if mode == ModeKeys.TRAIN:
            tvars = tf.trainable_variables()
            train_op, grads = clipped_train_op(loss, tvars, params)

        # Add summaries of helpful values.
        if mode == ModeKeys.TRAIN:
            tf.summary.scalar('loss', loss)
            for var in tvars:
                tf.summary.histogram(var.name, var)

            grads = list(zip(grads, tvars))
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            tf.summary.histogram('logits', logits)
            tf.summary.scalar('rms_rnn_activation',
                    tf.sqrt(tf.reduce_mean(output *
                output)))
            tf.summary.histogram('rnn_activations', output)
            # Argmax token distribution vs. target token distribution
            tf.summary.histogram('predicted_token', predictions['tokens'])
            tf.summary.scalar('argmax_bias', tf.argmax(softmax_b))
            tf.summary.histogram('target_token', targets)

            # Fraction correct.
            flat_logits = tf.reshape(logits, [-1, params['vocab_size']])
            flat_targets = tf.reshape(targets, (-1,))
            for k in [1, 5, 10]:
                at_k = tf.to_float(tf.nn.in_top_k(flat_logits, flat_targets,
                    k))
                tf.summary.scalar('hit@%d' % k, tf.reduce_mean(at_k))

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
    data_path = "/home/kevin/data/sanity-check/"

    BATCH_SIZE = 32
    UNROLL_LEN = 10

    train, valid, test, vocab = reader._raw_data(data_path)

    params = {}
    params['vocab_size'] = vocab.size()
    params['embedding_dim'] = 100
    params['batch_size'] = BATCH_SIZE
    params['unroll_length'] = UNROLL_LEN
    params['num_layers'] = 2
    params['learning_rate'] = 0.0001

    x, y = reader.example_producer(train, BATCH_SIZE, UNROLL_LEN)
    model = LanguageModel(params)
    z = model.model_fn(x, y, tf.contrib.learn.ModeKeys.TRAIN, params)
    init_op = tf.global_variables_initializer()

    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init_op)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        feed_dict = {}
        feed_dict[model.initial_state] = (np.random.randn(BATCH_SIZE,
                                          params['embedding_dim']),)*2*params['num_layers']
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
