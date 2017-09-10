import numpy as np
import itertools

def take(iterable, k):
    return list(itertools.islice(iterable, k))

def reservoir_sample(iterator, k):
    """ Samples k elements uniformly from iterator. """
    r = take(iterator, k)
    for i, x in enumerate(iterator):
        j = np.random.randint(0, i+k)
        if j < k:
            r[j] = x
    return r


def sample_errors(example_iterator, is_error_fn, N=100, is_shuffled=False):
    """ Returns |N| elements of example_iterator that satisfy |is_error_fn|.

    Args:
        example_iterator: iterable of dictionary of numpy arrays.
        is_error_fn: function (dict of ndarray) -> bool. Returns true if
                     example is an error.
        N: number of examples to draw.
        is_shuffled: if true, assumes example iterator is pre-shuffled.
    """
    errors = itertools.ifilter(is_error_fn, example_iterator)
    if is_shuffled:
        return take(errors, N)
    else:
        return reservoir_sample(errors, N)
