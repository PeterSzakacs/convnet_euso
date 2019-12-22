FROM rootproject/root-fedora

# setup install directory (to support tensorflow installs also from wheel files under docker/ directory)
ARG install_dir=/opt/convnet_euso
RUN mkdir $install_dir
WORKDIR $install_dir

# install required packages
RUN pip3 install scikit-learn scikit-image tflearn

# install tensorflow (default: latest CPU-only version)
ARG tf_version=tensorflow
COPY ./docker $install_dir/docker
RUN pip3 install $tf_version

# copy all sources and configurations to /opt
COPY ./src $install_dir/src
COPY ./config $install_dir/config

# setup wrappers in /usr/bin for all python scripts/tools
RUN docker/internal/setup.sh $install_dir
