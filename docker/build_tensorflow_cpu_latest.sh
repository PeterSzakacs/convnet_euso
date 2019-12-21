#!/bin/bash

# default: build with the latest version of CPU-only enabled tensorflow
# (requires a reasonably modern x86 CPU supporting at least AVX instructions)
cd ..
docker build -t peterszakacs/convnet_euso -f .