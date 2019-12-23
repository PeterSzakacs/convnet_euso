Docker build related tools/scripts.

This directory contains convenience scripts used for building docker images of this project with supported/tested/known good configurations, mainly in relation to Tensorflow versions. The `internal/` directory is for housing scripts used internally by the docker build to setup the environment after copying the project files to the image and for entrypoints.

# Building images

## Supported builds

For supported builds, we provide convenience shell scripts to invoke the build and appropriately tag the created image (these scripts all have names which start with `build_`).

All scripts for creating supported docker builds must be run within this directory. They support building the project with built in Python3-enabled CERN ROOT and 3 different versions of Tensorflow:

- Latest GPU-enabled version (requires a device and driver supporing at least CUDA 10.0)
- Latest CPU-only version (requires x86 CPU with AVX instruction support)
- CPU-only version 1.5.0 (for older x86 CPUs - requires at least SSE3 instruction support)

The CPU only versions are intented for use by developers in further development/maintenenace of this project. The GPU-enabled version is intended for actual production use.

## Customizing build

The Tensorflow package to install is configurable via a Docker build arg (`tf_version`) during build time. Thus, if any of the supported builds do not meet your requirements, you can create your own image with a custom version of Tensorflow (either pre-built or from a wheel file for e.g. builds from source on the target system) by running the following command from the project root:

```bash
$ docker build -t <optional_tag> --build_args tf_version=<custom_version> .
```

where `tf_version` must be of the form:

- `tensorfow==<version_number>` for prebuilt CPU-only Tensorflow packages
- `tensorfow-gpu==<version_number>` for prebuilt GPU-enabled Tensorflow packages
- `wheels/path/to/your/wheel/file` for custom Tensorflow builds from source (note that the wheel file to install must be located under the `wheels/` directory)

Do note that any custom Tensorflow builds must be built against the parameters of the image (Python 3.6 on latest [rootproject/root-fedora](https://github.com/root-project/root-docker/tree/master/fedora) image).

# Using the created images

The recommended way to use the tools provided in the images is to launch a new bash session within a running container where all tools are available. All data on which you wish to work should ideally be under a single directory on the system which can be mounted into the filesystem of the container running the image as a bind mount. 

To ease the verbosity of commands used to launch such a configuration, we provide also a convenience script which only requires the image name as its argument. The user and group id are set to those of the calling user and the users home directory is mounted into the container under the same path. Do note however that this user does not have a username within the container, which can lead to unexpected behavior.

All tools in the image are available via wrapper bash scripts put on directories in the system PATH, similar to standard command-line tools, so there is no need to invoke the python interpreter with the script file, e.g. instead of:

```bash
$ python3 /path/to/dataset_merger.py -h
```

you can simply use

```bash
$ dataset_merger -h
```

# Future work

Because of the verbosity of the commands used to build as well as run the images, support will later be added for docker-compose to simplify the command line invocations necessary for both actions (instead of just convenience scripts).
