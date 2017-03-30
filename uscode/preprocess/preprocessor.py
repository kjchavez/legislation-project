from tokenizer import tokenize
import glob
from constants import *
from vocab import Vocabulary


def _text_to_word_ids(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(t) for t in tokens]


def _file_to_word_ids(filename, vocab, sof_token=START_OF_FILE,
                      eof_token=END_OF_FILE):

    with open(filename) as fp:
        word_ids = _text_to_word_ids(fp.read(), vocab)

    return [vocab.get(sof_token)] + word_ids + [vocab.get(eof_token)]


class Preprocessor(object):
    def __init__(self, vocab_filename):
        self.vocab = Vocabulary(vocab_filename)

    def word_ids(self, text):
        return _text_to_word_ids(text, self.vocab)

    def title_to_word_ids(self, title_filename):
        return _file_to_word_ids(title_filename, self.vocab,
                                 sof_token=START_OF_TITLE,
                                 eof_token=END_OF_TITLE)


    def titles_to_word_ids(self, filepattern):
        ids = []
        for filename in glob.glob(filepattern):
            ids.extend(_file_to_word_ids(filename, self.vocab,
                                         sof_token=START_OF_TITLE,
                                         eof_token=END_OF_TITLE))

        return ids
