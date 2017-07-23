"""
    Creates the train/valid/test splits for a dataset.

    The situation:
        1. We have a way to iterate over all elements of the dataset, one by one.
        2. The entire dataset doesn't fit on disk.
        3. We're content to just extract some summary f(x) of the data point x.
        4. {f(x) : x in dataset } *does* fit on disk.


    This file helps us create those data sets.
"""
from __future__ import print_function

import logging
import time

class Feature(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def f_train(self, x):
        raise NotImplementedError

    def f_infer(self, x):
        return self.f_train(x)

def extract_features(example, features, attr):
    x = {}
    for f in features:
        try:
            x[f.name] = getattr(f, attr)(example)

        except Exception as err:
            logging.info("Error in %s: %s. Using default.",  f.name, str(err))
            x[f.name] = f.default

    return x

def extract_train_features(example, features):
    return extract_features(example, features, 'f_train')

def extract_infer_features(example, features):
    return extract_features(example, features, 'f_infer')


def process_data(generator, feature_extractors, filename,
                 max_elem=None, formatter=str):
    latest_update = time.time()
    with open(filename, 'w') as fp:
        for i, x in enumerate(generator):
            if max_elem and i == max_elem:
                logging.info("|max_elem|=%d reached.", max_elem)
                break
            currtime = time.time()
            if currtime - latest_update > 10:
                logging.info("Processed %d examples.", (i+1))
                latest_update = currtime
            fx = extract_train_features(x, feature_extractors)
            print(formatter(fx), file=fp)
