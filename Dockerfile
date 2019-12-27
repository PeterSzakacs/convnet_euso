FROM rootproject/root-fedora

# setup install directory (to support tensorflow installs also from wheel files under docker/ directory)
ARG install_dir=/opt/convnet_euso
RUN mkdir $install_dir
WORKDIR $install_dir

# install required packages
RUN pip3 install scikit-learn scikit-image tflearn

# install tensorflow (default: latest CPU-only version)
ARG tf_version=tensorflow
COPY ./wheels $install_dir/wheels
RUN pip3 install $tf_version

# copy all sources and configurations to /opt
COPY ./docker $install_dir/docker
COPY ./config $install_dir/config
COPY ./src $install_dir/src

# setup wrappers in /usr/bin for all python scripts/tools
RUN docker/internal/setup.sh $install_dir
ENV install_location=$install_dir
ENTRYPOINT ["bash", "-c", "${install_location}/docker/internal/entrypoint.sh"]
