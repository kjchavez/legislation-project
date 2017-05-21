# U.S. Legislation

Exploratory project using data from:

* U.S. legal code
* ProPublica Congress API
* congress.gov

See also: [`congressapi`](https://github.com/kjchavez/congressapi)

## Related Work

* https://www.aclweb.org/anthology/D/D16/D16-1221.pdf
* Political Science Dept. at MIT: http://cwarshaw.scripts.mit.edu/papers/CandidatePositions160810.pdf

## Build language model

### Generate dataset
Use congressdb to build a dataset of the bills introduced by the House in the 114th congress:

```
python -m congressdb.build --src=/data/congress --output=house-introduced-114 \
                             --type=hr --version=is --congress=114
```

This will create a directory `house-introduced-114` with training, validation, and test data splits and a vocab file. See `congressdb/build.py` for more info.

### Train model

A known set of good hyperparameter are in `hparams.yaml`.

```
python -m lm.train --data_path=house-introduced-114 --model_dir=/tmp/house-model --hparams=hparams.yaml
```

This will start training a model using the dataset we just generated. Snapshots of the model will go to `/tmp/house-model`.
To visualize/monitor the training process, start TensorBoard pointed at the model directory.

```
tensorboard --logdir=/tmp/house-model
```


### Sample some generated text

To create a sample from the language model, using the latest snapshot:

```
python -m lm.generate --model_dir=/tmp/house-model --data_path=house-introduced-114 \
                          --hyperparams=hparams.yaml --max_length=1000 --temp=1.1 \
                          --output=sample.txt
```

Omitting the `--output` flag will print to stdout.

