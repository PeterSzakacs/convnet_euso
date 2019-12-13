FROM rootproject/root-fedora

# install additional required packages
RUN pip3 install --upgrade scikit-learn scikit-image tflearn tensorflow

# copy all sources to /opt
ARG OUTPUT_DIR=/opt/convnet_euso
COPY . $OUTPUT_DIR

# dataset scripts
RUN $OUTPUT_DIR/docker_install.sh $OUTPUT_DIR
#RUN ln -s /usr/bin/dataset_generator.py  $OUTPUT_DIR/src/dataset_generator.py
#RUN ln -s /usr/bin/dataset_shuffler.py  $OUTPUT_DIR/src/dataset_shuffler.pys
#RUN ln -s /usr/bin/dataset_condenser.py  $OUTPUT_DIR/src/dataset_condenser.py
#RUN ln -s /usr/bin/dataset_meta_value_distribution.py  $OUTPUT_DIR/src/dataset_meta_value_distribution.py
#RUN ln -s /usr/bin/dataset_merger.py  $OUTPUT_DIR/src/dataset_merger.py
#RUN ln -s /usr/bin/dataset_visualizer.py  $OUTPUT_DIR/src/dataset_visualizer.py

# network model scripts
#RUN ln -s /usr/bin/model_xvalidator.py  $OUTPUT_DIR/src/model_xvalidator.py
#RUN ln -s /usr/bin/model_trainer.py $OUTPUT_DIR/src/model_trainer.py
#RUN ln -s /usr/bin/model_checker.py $OUTPUT_DIR/src/model_checker.py
#RUN ln -s /usr/bin/model_activations_visualizer.py  $OUTPUT_DIR/src/model_activations_visualizer.py
#RUN ln -s /usr/bin/convnet_filter_visualizer.py $OUTPUT_DIR/src/convnet_filter_visualizer.py

# evaluation result processing scripts
#RUN ln -s /usr/bin/eval_to_sensitivity_plot.py  $OUTPUT_DIR/src/eval_to_sensitivity_plot.py
#RUN ln -s /usr/bin/eval_to_html.py  $OUTPUT_DIR/src/eval_to_html.py
