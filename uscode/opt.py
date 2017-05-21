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

def create_optimizer(class_name, params):
    cls = _optimizer_class(class_name)
    instance = cls(**params)
    return instance

def load_optimizer(params_file):
    class_name, params = _opt_params(params_file)
    return create_optimizer(class_name, params)

def _constructor_args(cls):
    argspec = inspect.getargspec(cls.__init__)
    return [arg for arg in argspec[0] if arg != 'self']
