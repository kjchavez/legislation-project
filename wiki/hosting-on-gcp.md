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
