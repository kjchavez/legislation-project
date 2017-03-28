from tokenize import tokenize
import glob
from preprocess.constants import *
from vocab import Vocabulary


def _text_to_word_ids(text, vocab):
    return [vocab.get(t) for t in tokenize(text)]


def _file_to_word_ids(filename, vocab, sof_token=START_OF_FILE,
                      eof_token=END_OF_FILE):

    with open(filename) as fp:
        word_ids = _text_to_word_ids(fp.read())

    return [vocab[sof_token]] + word_ids + [vocab[eof_token]]


class Preprocessor(object):
    def __init__(self, vocab_filename):
        self.vocab = Vocabulary(vocab_filename)

    def word_ids(self, text):
        return _text_to_word_ids(text, self.vocab)

    def title_to_word_ids(self, title_filename):
        return _file_to_word_ids(title_filename, self.vocab,
                                 sof_token=START_OF_TITLE,
                                 eof_token=END_OF_TITLE)


    def filepattern_to_word_ids(self, filepattern):
        ids = []
        for filename in glob.glob(filepattern):
            ids.extend(_file_to_word_ids(filename, vocab,
                                         sof_token=sof_token,
                                         eof_token=eof_token))

        return ids



