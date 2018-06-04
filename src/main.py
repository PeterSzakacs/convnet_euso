# from __future__ import division, print_function, absolute_import

# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.normalization import local_response_normalization
# from tflearn.layers.estimator import regression
# import event_visualization as eviz

import numpy as np
import csv, itertools
import sys
# import matplotlib.pyplot as plt
# import matplotlib.animation as anim

baseDir = "/media/szakacs/Seagate/Large/Diplomovka/data"
files = []
with open("visible_events.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        files.append(baseDir + "/" + row['source_file_acquisition'])

X = np.ndarray(shape=(2 * len(files), 48, 48))
Y = []
idx=0
total_size = 0

for filename in files:
    print("loading file " + filename)
    event_frames = np.load(filename)
    orig_max_x_y_arr   = np.max(event_frames, axis=0)
    first_frame = event_frames[0]
    X[idx] = orig_max_x_y_arr
    X[idx+1] = first_frame
    idx += 2
    Y.append(1)
    Y.append(0)
    total_size += event_frames[0].nbytes + orig_max_x_y_arr.nbytes

print("Estimated RAM used: " + str((total_size) / 1024 / 1024) + " MiB")
print("Estimated RAM used 2: " + str(X.nbytes / 1024 / 1024) + " MiB")
print("read all files")

np.save(baseDir + "/visible_events", X)


# Building convolutional network
# network = input_data(shape=[None, 48, 48, 1], name='input')
# network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 2)
# network = local_response_normalization(network)
# network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 2)
# network = local_response_normalization(network)
# network = fully_connected(network, 128, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 256, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 10, activation='softmax')
# network = regression(network, optimizer='adam', learning_rate=0.01,
#                      loss='categorical_crossentropy', name='target')
#
# # Training
# model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit({'input': X}, {'target': Y}, n_epoch=1,
#            validation_set=({'input': testX}, {'target': testY}),
#            snapshot_step=100, show_metric=True, run_id='convnet_mnist')





















# event_frames = np.load('./res/ev_98_mc_1__signals_p128_a0_g30_f128_b20170502-124722-001.001_k1_s0_d32_n11_m128.npy')

# orig_max_x_y_arr   = np.max(event_frames, axis=0)
# orig_max_gtu_y_arr = np.transpose(np.max(event_frames, axis=2))
# orig_max_gtu_x_arr = np.transpose(np.max(event_frames, axis=1))

# fig, ax = eviz.visualize_frame(event_frames[0])

# eviz.visualize_frame(orig_max_x_y_arr)
# eviz.visualize_frame_gtu_x(orig_max_gtu_x_arr[0:, 150:200])
# eviz.visualize_frame_gtu_y(orig_max_gtu_y_arr[0:, 150:200])



# eviz.visualize_frame_gtu_x(event_frames[0])
# eviz.visualize_frame_gtu_y(event_frames[0])
# eviz.visualize_frame_num_relation(event_frames[0])

# eviz.visualize_hough_lines(orig_max_x_y_arr, np.arange(10))

# fig, ax = plt.subplots()
#
# frames = []
# for x in range(0, np.size(event_frames, axis=0)):
#     im = ax.imshow(event_frames[x], animated=True)
#     frames.append([im])
#
# animation = anim.ArtistAnimation(fig, frames, blit=True, repeat_delay=2000)
