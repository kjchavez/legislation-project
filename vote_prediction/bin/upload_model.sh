#!/usr/bin/env bash

if [ $# -eq 0 ]
  then
    echo "Please provide model name."
    exit
fi

MODEL_NAME=$1
GCLOUD_SAFE_MODEL_NAME=${MODEL_NAME//-/_}

# Set up on GCP. Doesn't necessarily need to change, even if I change the
# I/O signature of the model.
GCLOUD_MODEL="vote_prediction"

# This is my bucket.
GCLOUD_BUCKET="us-legislation-models"

# And a subdirectory I created.
PROJECT_FOLDER="vote-prediction-naive"

# Extract the latest version (names are timestamps)
VERSION="$(ls -1 export/${MODEL_NAME} | tail -1)"
MODEL_VERSION_ID="${GCLOUD_SAFE_MODEL_NAME}_${VERSION}"

# Push the model to Google Cloud Storage and upload a new model on Cloud ML Engine.
gsutil cp -r export/${MODEL_NAME}/${VERSION} gs://${GCLOUD_BUCKET}/${PROJECT_FOLDER}/${MODEL_VERSION_ID} &&
gcloud ml-engine versions create ${MODEL_VERSION_ID} --model=${GCLOUD_MODEL} --origin=gs://${GCLOUD_BUCKET}/${PROJECT_FOLDER}/${MODEL_VERSION_ID} --runtime-version=1.2
echo "Uploaded version:"
echo ${MODEL_VERSION_ID}
