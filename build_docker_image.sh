#!/usr/bin/env bash

REPO=opera
IMAGE=rtc
TAG=0.2_beta

echo "IMAGE is $REPO/$IMAGE:$TAG"

# fail on any non-zero exit codes
set -ex

# Not sure how we are using sdist in this script...
python3 setup.py sdist

# build image
docker build --rm --force-rm --network host -t $REPO/$IMAGE:$TAG -f Docker/Dockerfile .

# run tests - to be worked on when the RTC test module is in place
#docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/mnt" -w /mnt -it --network host "${IMAGE}:$t" pytest /mnt/tests/

# create image tar
docker save $REPO/$IMAGE:$TAG > Docker/dockerimg_rtc_beta_0.2.tar

# remove image
docker image rm $REPO/$IMAGE:$TAG
