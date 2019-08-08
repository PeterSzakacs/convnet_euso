PIP Package lists with known good/tested sets of dependencies for this project.

Tested on Python 3.5/Ubuntu 16.04 and Python 3.6/Ubuntu 18.04.

#Conventions

Package lists here are grouped primarily by Python 3 major versions (e.g. 3.5, 3.6, etc.), For all versions, there are 
3 package lists with the following :

- base - base packages other than Tensorflow and CERN ROOT, required for tools/scripts working with datasets (except dataset_condenser)
- tensorflow-cpu - packages for the CPU version of tensorflow (for CPUs supporting at least SSE 3.2 instructions)
- tensorflow-gpu - packages for the GPU version of tensorflow (for GPUs supporting at least CUDA compute API v6.1)

To install/setup the project dependencies (minus CERN ROOT) in a virtualenv, after setting the environment by 
`source`-ing it, install by running:

```bash
$ pip install -r python<version>-requirements-base.txt -r python<version>-requirements-tensorflow-<cpu|gpu>.txt
```