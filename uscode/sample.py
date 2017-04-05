import numpy as np

def sample(distribution, temp=1.0):
    """ Samples from a distribution with a given temperature.

    Args:
        distribution: an M x N numpy array representing M distributions over N
                      possible values.
    """
    assert temp >= 0.1, "Temp must be at least 0.1"
    M, N = distribution.shape
    coef = 1.0 / temp
    dist = np.power(distribution, coef)
    dist /= np.sum(dist, axis=1, keepdims=True)
    values = np.empty(M)
    for i in xrange(M):
        values[i] = np.random.choice(xrange(N), p=dist[i, :])

    return values
