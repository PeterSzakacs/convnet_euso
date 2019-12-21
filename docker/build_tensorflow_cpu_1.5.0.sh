#!/bin/bash

# build with version 1.5.0 of CPU-only enabled tensorflow
# (requires x86 CPUs supporting at least SSE 3 instruction set)
cd ..
docker build --build-arg tf_version=tensorflow==1.5.0 -t peterszakacs/convnet_euso_cpu_1.5.0 .