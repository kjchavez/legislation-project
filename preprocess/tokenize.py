# coding=utf-8

""" Function to 'clean' and tokenize a string for use in legislation-project
    models.

    This function should be used both to create datasets and at inference time.
"""
import re
from nltk import word_tokenize

def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


REPLACEMENTS = {
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
}

def tokenize(string):
    return word_tokenize(multireplace(string, REPLACEMENTS).decode('utf-8'))
