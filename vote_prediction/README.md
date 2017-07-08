# Vote Prediction

## Task

> Given a member of Congress and a legislative proposal, predict whether they vote `Aye` or not.

**Some important notes:**

* We might have **no** historical data for that Congress member - for example, it's his/her first time in office. However, the model should still be able to provide a best estimate. 
* 

## Data

The GovTrack dataset contains *roll call* vote data.

We're only interest in votes on the **passage** of **bills** or **joint resolutions**. These are legislative proposals, which -- if passed -- have the force of law. Congress votes on a variety of other things (e.g. rules governing their internal processes) which we are not considering in this task.
Some exploration of this data in `analysis/Votes.ipynb`.

### Quick Stats ###

* Spans more than 3500 roll calls.
* Contains over 1 million individual votes.
* 70.8% of the votes = Aye
* 25.6% of the votes = Nay
* 3.5% of the votes = Not Voting
* <0.1% of the votes = Present


### Partitioning data

We split the 3500 roll call votes into train/validation/test votes at ratios of 0.7 / 0.1 / 0.2.
Running `create_dataset.py` will create these splits and record which bill is in which split in `data/METADATA`


## Baseline models

A somewhat cynical, partisan, rule-based model would be...

> If the sponsor and voter are in the same party, then predict they vote `aye`.

With this we get ~75% accuracy. So this is the very base, the floor.

## Upload model

```bash
gsutil cp -r export/1234565 gs://[DESTINATION_BUCKET_NAME]/vote-prediction-naive/$VERSION_NAME
```

Add it in GCP console.

Test to make sure it worked:

```
gcloud ml-engine predict --model vote_prediction --version $VERSION_NAME --json-instances test/request.json
```
