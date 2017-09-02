import tensorflow as tf

def optimize_and_monitor(optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars,
                                         global_step=tf.train.get_or_create_global_step())
    # Add summary of weights and gradients.
    for grad, var in grads_and_vars:
        try:
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradient', grad)
        except:
            print("Failed to add summary for" + var.name)

    return train_op


def add_binary_classification_metrics(predictions, labels, eval_metrics):
    eval_metrics['accuracy'] = tf.metrics.accuracy(tf.equal(labels, 1), predictions)
    eval_metrics['precision'] = tf.metrics.precision(labels, predictions)
    eval_metrics['recall'] = tf.metrics.recall(labels, predictions)
