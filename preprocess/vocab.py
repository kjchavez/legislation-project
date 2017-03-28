class Vocabulary(object):
    UNKNOWN = "<UNK>"

    def __init__(self, vocab_filename):
        self.word_to_id = {}
        self.word_to_id[Vocabulary.UNKNOWN] = 0
        idx = 1
        with open(vocab_filename) as fp:
            for line in fp:
                word = line.split()[0]
                self.word_to_id[word] = idx
                idx += 1

    def get(self, token):
        """ Returns id of token, or id for <UNK> if token is unknown. """
        return self.word_to_id.get(token,
                self.word_to_id[Vocabulary.UNKNOWN])

    def size(self):
        return len(self.word_to_id)
