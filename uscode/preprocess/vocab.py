import codecs
import collections
import glob
import unicodedata
import sys

from nltk.tokenize import wordpunct_tokenize


def _freq(key, counter):
    return float(counter[key]) / sum(counter.values())

def _file_contents_generator(filepattern):
    for filename in glob.glob(filepattern):
        with open(filename) as fp:
            yield filename, fp.read()

class Vocabulary(object):
    OUT_OF_VOCABULARY = "<oov>"

    def __init__(self, tokens):
        self.word_to_id = {}
        self.id_to_word = [Vocabulary.OUT_OF_VOCABULARY]
        self.word_to_id[Vocabulary.OUT_OF_VOCABULARY] = 0
        idx = 1
        for token in tokens:
            self.word_to_id[token] = idx
            self.id_to_word.append(token)
            idx += 1

    def saveto(self, filename):
        with codecs.open(filename, 'w', encoding='utf-8') as fp:
            for word in self.id_to_word:
                try:
                    fp.write(word)
                except:
                    print "Error in saving vocabulary."
                    print "Offending token:", word
                    print "Individual chars:", [c for c in word]
                    print "If tokens are coming from file, check encoding!" 
                    sys.exit(1)

                fp.write('\n')


    def get(self, token):
        """ Returns id of token, or id for <oov> if token is unknown. """
        return self.word_to_id.get(token,
                self.word_to_id[Vocabulary.OUT_OF_VOCABULARY])

    def get_token(self, idx):
        return self.id_to_word[idx]

    def size(self):
        return len(self.word_to_id)

    @staticmethod
    def fromfile(filename):
        with codecs.open(filename, encoding='utf-8') as fp:
            oov = next(fp).strip()
            assert oov == Vocabulary.OUT_OF_VOCABULARY
            return Vocabulary([line.split()[0] for line in fp])

    @staticmethod
    def fromiterator(iterator, tokenize_fn=wordpunct_tokenize,
                     max_num_tokens=None, min_freq=None, min_count=None):
        """ Builds vocab from an iterator that yields (id, text) tuples."""
        counter = collections.Counter()
        for idx, text in iterator:
            print "Processing:", idx
            counter.update(tokenize_fn(text))

        tokens = []
        for key, count in counter.most_common(max_num_tokens):
            if (min_freq and _freq(key, counter) < min_freq) or \
               (min_count and count < min_count):
                break
            tokens.append(key)

        return Vocabulary(tokens)

    @staticmethod
    def build(filepattern, tokenize_fn, max_num_tokens, min_freq=None, min_count=None):
        return Vocabulary.fromiterator(_file_contents_generator(filepattern),
                                       tokenize_fn=tokenize_fn, min_freq=min_freq,
                                       min_count=min_count)

