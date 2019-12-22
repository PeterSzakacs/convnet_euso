#!/bin/bash

# build with the latest GPU version of tensorflow
# (requires a device supporting at least CUDA 10.0 API)
cd ..
docker build --build-arg tf_version=tensorflow-gpu -t peterszakacs/convnet_euso_gpu .