import codecs
import collections
import glob


def _freq(key, counter):
    return float(counter[key]) / sum(counter.values())


class Vocabulary(object):
    OUT_OF_VOCABULARY = "<oov>"

    def __init__(self, tokens):
        self.word_to_id = {}
        self.id_to_word = []
        self.word_to_id[Vocabulary.OUT_OF_VOCABULARY] = 0
        idx = 1
        for token in tokens:
            self.word_to_id[token] = idx
            self.id_to_word.append(token)
            idx += 1

    def saveto(self, filename):
        with codecs.open(filename, 'w', encoding='utf-8') as fp:
            for word in self.id_to_word:
                fp.write(word)
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
        with open(filename) as fp:
            return Vocabulary([line.split()[0] for line in fp])

    @staticmethod
    def build(filepattern, tokenize_fn, max_num_tokens, min_freq=None, min_count=None):
        counter = collections.Counter()
        for filename in glob.glob(filepattern):
            print "Processing:", filename
            with open(filename) as fp:
                for line in fp:
                    counter.update(tokenize_fn(line))

        tokens = []
        for key, count in counter.most_common(max_num_tokens):
            if (min_freq and _freq(key, counter) < min_freq) or \
               (min_count and count < min_count):
                break
            tokens.append(key)

        return Vocabulary(tokens)

