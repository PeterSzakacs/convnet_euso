Tools for working with EUSO-SPB mission data and training neural networks for classification of aforementioned data.

Tested on Python 3.5/Ubuntu 16.04 and Python 3.6/Ubuntu 18.04.

#Project Setup

To get started, create a new Python virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) 
or the [`venv`](https://docs.python.org/3/library/venv.html) module of Python 3.5+. Although all dependencies can be 
installed system-wide, this is not recommended.

Under the `venv/` directory can be found maintained lists of required packages for all components in this project to run. 
These represent known working sets of packages of this project (as of 07/2019). Further instructions for their use are 
in the README for that directory.

Alternatively, simply install the following pip packages:

- Numpy
- Matplotlib
- Scikit-image
- Scikit-learn
- TFlearn
- Pandas

For either method, a special case is CERN ROOT with Python 3 bindings and Tensorflow, both described below.

## Tensorflow

The required version of Tensorflow to install depends on the target system used for running this project, specifically 
the device used for training the network(s). If a CUDA device is present (preferred), install a suitable version of 
`tensorflow-gpu`, else install a suitable version of `tensorflow`. 

The default binaries are precompiled to take advantage of specific instructions on the target system (processor ISA 
extensions, e.g. SSE, AVX and CUDA Compute API versions). For systems with an older CPU or GPU - or driver - the latest 
version may thus not be suitable. 

The `venv/` directory contains package lists for both GPU and CPU versions of Tensorflow, and both divided into 
Python 3.5 and Python 3.6-compatible/tested lists. The CPU versions have been tested to work on Any CPU supporting 
at least SSE 3.2. The CPU versions however, are intended more for development of the tools in this project, particularly 
those for working with neural networks, and not suitable for training due to low throughput during training on CPU.

A workaround for getting the most optimized version of Tensorflow for the target system is to [compile from source](https://www.tensorflow.org/install/source), 
but we haven't seriously tested this option.

## CERN ROOT

The default binaries for CERN ROOT do not contain Python 3 bindings as of the time of writing this guide. As a result, 
the preferred method to setup ROOT is to compile from source following the instructions at the 
[project website](https://root.cern.ch/building-root).

Apart from CMake and a suitable C++ compiler (e.g. g++) and build tool (e.g. Make), for Ubuntu and derived systems, 
the following packages are also required.

`libx11-dev libxpm-dev libxft-dev libxext-dev libpython3-dev`

Though not strictly necessary, the commands for configuring and actually running the build should be run in a python 
virtualenv shell session. Cmake will then detect the packages installed in the virtualenv (particularly NumPy) and 
incorporate them during the build process.

After downloading ROOT sources (archive or checked out from git), create a build directory and run the following 
command in it:

```bash
~$ cmake -Dpython3=ON /path/to/sourcedir
```

Once the configuration has completed (check CMake logs to the console to verify python 3 support has been added), run 
the following command to actually build ROOT.

```bash
~$ cmake --build /path/to/builddir  [-- <options to the native tool>]
```

Please note that the build may take a long time, depending on available system resources. An option to speed it up is 
to use multiple threads. For a build using Make, this is done by specifying the `-j NumProcs` as a build option for 
the native tool.

After the build finishes, the ROOT binaries and libraries can be imported into the current shell session by 
`source`-ing the build directory (this does not influence the previous environment set by `source`-ing the virtualenv).

```bash
~$ source /path/to/builddir/bin/thisroot.sh
```

In this configuration, the build directory is effectively also the install directory.