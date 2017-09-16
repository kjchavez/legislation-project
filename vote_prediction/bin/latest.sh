#!/usr/bin/env bash

MODEL_NAME=$1
VERSION="$(ls -1 export/${MODEL_NAME} | tail -1)"

echo "export/${MODEL_NAME}/${VERSION}"
