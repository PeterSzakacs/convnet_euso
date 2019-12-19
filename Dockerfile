FROM rootproject/root-fedora

# install additional required packages
RUN pip3 install --upgrade scikit-learn scikit-image tflearn tensorflow

# copy all sources to /opt
ARG OUTPUT_DIR=/opt/convnet_euso
COPY . $OUTPUT_DIR

# dataset scripts
RUN $OUTPUT_DIR/docker_install.sh $OUTPUT_DIR
