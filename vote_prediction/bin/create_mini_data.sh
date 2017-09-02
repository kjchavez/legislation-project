#!/usr/bin/env bash

python create_dataset.py --output=data/votes-mini.dat --nmax=1000
shuf data/votes-mini.dat -o data/votes-mini.dat
python tfrecord_util.py --data=data/votes-mini.dat --outdir=mini-data --text_fields=BillTitle
