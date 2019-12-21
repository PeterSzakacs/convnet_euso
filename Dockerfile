FROM rootproject/root-fedora

# install additional required packages
RUN pip3 install --upgrade scikit-learn scikit-image tflearn

# by default, use the latest CPU version of tensorflow
ARG tf_version=tensorflow
RUN pip3 install --upgrade $tf_version

# copy all sources to /opt
ARG OUTPUT_DIR=/opt/convnet_euso
COPY . $OUTPUT_DIR

# setup wrappers in /usr/bin for all python scripts/tools
RUN $OUTPUT_DIR/docker_setup.sh $OUTPUT_DIR
