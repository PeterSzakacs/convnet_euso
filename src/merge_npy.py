import os
import sys

import csv
import numpy as np

import utils.cmd_interface_npymerger as cmd

# script to coallesce visible event frames from several .npy files
# into a single larger .npy file for easier loading and processing
# by the network.

cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])

baseDir = args.srcdir
files = []
with open(args.infile) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        files.append(os.path.join(baseDir, row['source_file_acquisition']))

X = np.ndarray(shape=(2 * len(files), 48, 48))
Y = np.empty(shape=(2 * len(files), 2))
idx=0
#total_size = 0

for filename in files:
    print("loading file " + filename)
    event_frames = np.load(filename)
    orig_max_x_y_arr = np.max(event_frames[155:175], axis=0)
    noise_frame = np.max(event_frames[0:20], axis=0)
    X[idx] = orig_max_x_y_arr
    X[idx+1] = noise_frame
    np.put(Y[idx], [0, 1], [1, 0])
    np.put(Y[idx+1], [0, 1], [0, 1])
    idx += 2
#    total_size += event_frames[0].nbytes + orig_max_x_y_arr.nbytes

#print("Estimated RAM used: " + str((total_size) / 1024 / 1024) + " MiB")
#print("Estimated RAM used 2: " + str(X.nbytes / 1024 / 1024) + " MiB")
print("read all files")


np.save(args.outfile, X)
np.save(args.targetfile, Y)
