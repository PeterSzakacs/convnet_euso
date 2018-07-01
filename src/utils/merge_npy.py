import numpy as np
import csv

# script to coallesce visible event frames from several .npy files
# into a single larger .npy file for easier loading and processing
# by the network.

# TODO: Make baseDir a command line argument
baseDir = "/media/szakacs/Seagate/Large/Diplomovka/data"
files = []
with open("visible_events.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        files.append(baseDir + "/" + row['source_file_acquisition'])

X = np.ndarray(shape=(2 * len(files), 48, 48))
Y = np.tile([1,0], [int(X.shape[0]/2)])
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
    total_size += event_frames[0].nbytes + orig_max_x_y_arr.nbytes

print("Estimated RAM used: " + str((total_size) / 1024 / 1024) + " MiB")
print("Estimated RAM used 2: " + str(X.nbytes / 1024 / 1024) + " MiB")
print("read all files")


output_dir = "res"
np.save(output_dir + "/visible_events_x", X)
np.save(output_dir + "/visible_events_y", Y)