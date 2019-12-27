#!/bin/bash

# convenience script to launch an interactive shell session in a new container
# with the given image as the calling user and a bind mount to the calling users home directory

if [ "$#" -lt 1 ]; then
  >&2 echo "Usage: $0 <image_name>"
  exit 1
fi

IMAGE_NAME=$1

docker run --user "$(id -u):$(id -g)" --workdir "$HOME" -v "$HOME:$HOME" --rm -it "$IMAGE_NAME" bash