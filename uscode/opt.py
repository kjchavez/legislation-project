"""
    Consistent mechanism for setting hyperparameters for optimization method.

    To define an optimizer via a YAML file of hyperparameters, hparams.yaml:

    opt_method: 'GradientDescentOptimizer'  (or any tf.train.Optimizer)
    opt_params:
      learning_rate: 0.001
      use_locking: true
      ... (other params here) ...

    Then, in code, load an instantiated optimizer:

    optimizer = opt.load_optimizer('hparams.yaml')

"""

import yaml
import inspect
import tensorflow as tf

def _opt_params(yaml_file):
    with open(yaml_file) as fp:
        params = yaml.load(fp)
    if 'opt_method' not in params:
        raise KeyError('opt_method not found in params file')
    if 'opt_params' not in params:
        raise KeyError('opt_params not found in params file')

    return params['opt_method'], params['opt_params']

def _optimizer_class(class_name):
    """ Returns class from tf.train module with the given |class_name|. """
    optimizer = getattr(tf.train, class_name)
    return optimizer

def _create_decaying_lr(lr_params):
    if 'decay_function' not in lr_params:
        raise KeyError('decay_function not found in lr_params')
    if 'decay_args' not in lr_params:
        raise KeyError('decay_args not found in lr_params')
    decay_fn = _optimizer_class(lr_params['decay_function'])
    args = lr_params['decay_args']

    # This arg should always come from the framework, not file.
    if 'global_step' in inspect.getargspec(decay_fn).args:
        args['global_step'] = tf.contrib.framework.get_or_create_global_step()
    return decay_fn(**lr_params['decay_args'])

def create_optimizer(class_name, params):
    # If the 'learning_rate' param is another dictionary of params,
    # assume we're creating a decaying learning rate.
    if 'learning_rate' in params:
        lr_params = params['learning_rate']
        if isinstance(lr_params, dict):
            print "Converted params to object"
            decaying_lr = _create_decaying_lr(lr_params)
            params['learning_rate'] = decaying_lr
        else:
            print "Not converting learning_rate of type: %s" % type(lr_params)

    cls = _optimizer_class(class_name)
    instance = cls(**params)
    return instance

def load_optimizer(params_file):
    class_name, params = _opt_params(params_file)
    return create_optimizer(class_name, params)

def _constructor_args(cls):
    argspec = inspect.getargspec(cls.__init__)
    return [arg for arg in argspec[0] if arg != 'self']
