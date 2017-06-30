#!/usr/bin/env python

import argparse
import os
from os.path import join, expanduser, exists
from subprocess import call

if __name__ != '__main__':
    raise ImportError("this module cannot be imported")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", '-t',
            default=join(expanduser('~'), 'govtrackdata'),
            help="Target directory for rsync.")
    return parser.parse_args()

args = parse_args()
if not exists(args.target):
    print "NOTICE: The first sync will take a considerable amount of time."
    print "It is about 100 GB of data. Consider a nice, long coffee break."
    os.makedirs(args.target)

subprocess.call(['rsync', '-avz', 'govtrack.us::govtrackdata', args.target])
