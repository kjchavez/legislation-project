#!/usr/bin/env bash

#python create_dataset.py --output=data/votes.dat
shuf data/votes.dat -o data/votes.dat
python tfrecord_util.py --data=data/votes.dat --outdir=data --text_fields=BillTitle
