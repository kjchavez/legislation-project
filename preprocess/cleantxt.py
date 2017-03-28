# coding=utf-8
import re
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepattern", '-f', default="uscode_text/*.txt")

    return parser.parse_args()


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



replacements = {
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
}


def main():
    args = parse_args()

    for filename in glob.glob(args.filepattern):
        with open(filename, 'r') as fp:
            clean = multireplace(fp.read(), replacements)

        with open(filename, 'w') as fp:
            fp.write(clean)


if __name__ == "__main__":
    main()
