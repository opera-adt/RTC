#!/usr/bin/env bash

REPO=opera
IMAGE=rtc
TAG=final_1.0.4

docker_build_args=(--rm --force-rm --network host -t $REPO/$IMAGE:$TAG -f docker/Dockerfile)

if [ $# -eq 0 ]; then
    echo "Base image was not specified. Using the default image specified in the Dockerfile."
else
    echo "Using $1 as the base image."
    docker_build_args+=(--build-arg BASE_IMAGE=$1)
fi

echo "IMAGE is $REPO/$IMAGE:$TAG"

# fail on any non-zero exit codes
set -ex

# Not sure how we are using sdist in this script...
python3 setup.py sdist

# build image
docker build "${docker_build_args[@]}" .
# run tests - to be worked on when the RTC test module is in place
#docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/mnt" -w /mnt -it --network host "${IMAGE}:$t" pytest /mnt/tests/

# create image tar
docker save $REPO/$IMAGE:$TAG > Docker/dockerimg_rtc_${TAG}.tar

# remove image
docker image rm $REPO/$IMAGE:$TAG
