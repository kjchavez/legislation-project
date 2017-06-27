# Hosting on Google Cloud Platform

For future reference. You're welcome future self ;)

We've trained a language model for bills introduced into the House of Representatives over the past 20+ years.
We can export the model and serve it locally with Tensorflow Serving. But, I don't *actually* want to serve the
model from my local machine.

Hopefully, serving the model from Google Cloud Platform should be painless. This page chronicles the journey.

[TOC]

## Resources

We're going to use the Cloud ML Engine. The documentation looks fairly extensive.

Here's a set of How-Tos from them:
https://cloud.google.com/ml-engine/docs/how-tos/

## Get Situated in Google Cloud Platform

Go to the developer console for GCP. Create a new project.

> NOTE: I don't know if I have payment information already added to my account.

* Back on the home page for the Console, make sure you switch to the right project! Drop down on the top navbar.
* Click on the ML Engine tab on the left panel. This starts *enabling* the Cloud ML engine API for the project. Says it could take up to 10 minutes. Not sure why.

> WARNING: If you haven't enabled billing, you'll have to do so before using the ML Engine. (As of 06/2017).

The [Getting Started](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction) overview seems like a good starting point.


## Command Line for Google Cloud

I'm running Ubuntu 14.04.5, (which unfortunately will reach end of life in April 2019) and followed the apt-get instructions on https://cloud.google.com/sdk/downloads.

Running `gcloud init` asks me a bunch of questions that I'm not sure I know the answer to. 
Enabled Compute Engine with region us-west-1a. Needed.
Set up Application Default Credentials.  tl;dr; user-independent auth credentials.

Seems like a lot of configuration will be saved in `/home/kevin/.config/gcloud/`

NOTE: Looks like the /usr/lib/google-cloud-sdk/bin/ wasn't permanently added to my path. Might want to edit bashrc manually.

## Export a SavedModel locally

See the `lm/export.py` file for an example. Doesn't matter how it's created as long as it has a `serving_default` serving signature.


## Some Gotchas About the Model
* Requires that data have **instance keys** on the input, which are passed through to the output unaltered. So that distributed processing works.
* If input has **binary data**, the input/output alias for that value has to end in **_bytes**. It will be base-64 encoded when the request/response goes over the network.


## Upload model to Google Cloud Storage

https://cloud.google.com/storage/docs/object-basics

1. Create a bucket for the project.

2. Then using `gsutil`

```bash
gsutil cp -r saved-models/[export_name]/1 gs://[DESTINATION_BUCKET_NAME]/[model_name]
```

## Create a Model on Cloud ML Engine

A *model* is just a container for *versions*. The versions are the actual models.
Create a version with the UI, point to the cloud storage version directory.

Query the version using `gcloud`.

```
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION_NAME --json-instances $INPUT_DATA_FILE
```

where the input data file for me looked like this:

```
{ "temp": 1.0  }
```

## Hosting the rest on App Engine

* Need to install app-engine-python. At least to have a python app.
* Make sure you're forwarding some port from your remote machine (if you have one) so the dev_appserver.py is actually useful.
* If using python, make sure to read the instructions about virtualenv and the appconfig file *carefully*: https://cloud.google.com/appengine/docs/standard/python/getting-started/python-standard-env 
* Relative URLs are important! Otherwise you might end up with the dreaded Access-Control-Allow-Origin error. There is a way around it by allowing cross-domain resource loading. I never quite figured out how.

## Questions
* Do I get charged per app deployment?
