from nltk.tokenize import wordpunct_tokenize

_TOKENIZERS = {}

def registered(fn):
    _TOKENIZERS[fn.func_name] = fn
    return fn


def _lower(tokens):
    return [w.lower() for w in tokens]


@registered
def wordpunct(text):
    return wordpunct_tokenize(text)

@registered
def wordpunct_lower(text):
    return _lower(wordpunct_tokenize(text))

def get_tokenizer(name):
    return _TOKENIZERS.get(name)

def tokenizer_names():
    return _TOKENIZERS.keys()

def get_name(fn):
    return fn.func_name

def is_registered_tokenizer(fn):
    return any(fn == registered_fn for registered_fn in _TOKENIZERS.values())
