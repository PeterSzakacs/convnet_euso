import os
import sys
import csv

import numpy as np

import utils.data_converters as conv
import utils.cmdint.cmd_interface_creator as cmd

# script to coallesce (simulated or real) data from several files 
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored


cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])

file_converter = None
if args.simu:
    def simu_dataconv(srcfile, triggerfile):
        data, targets = conv.simu_data_to_dataset(srcfile, triggerfile)
        num_data = len(data)
        metadata_dict = [{'source_file_acquisition_full': srcfile, 'packet_idx': 1}] * num_data
        return num_data, data, targets, metadata_dict
    file_converter = lambda srcfile, triggerfile: simu_dataconv(srcfile, triggerfile)
else:
    def flight_dataconv(srcfile, triggerfile):
        data, targets = conv.flight_data_to_dataset(srcfile, triggerfile)
        num_data = len(data)
        metadata_dict = [{'source_file_acquisition_full': srcfile, 'packet_idx': idx}
                         for idx in range(num_data)]
        return num_data, data, targets, metadata_dict
    file_converter = lambda srcfile, triggerfile: flight_dataconv(srcfile, triggerfile)

input_tsv = args.filelist
output_tsv = os.path.join(args.outdir, args.name + '_meta.tsv')

num_frames = 0
X, Y = [], []
with open(input_tsv, 'r') as infile, open(output_tsv, 'w') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    writer = csv.DictWriter(outfile, fieldnames=['source_file_acquisition_full', 'packet_idx'], delimiter='\t')
    writer.writeheader()
    for row in reader:
        srcfile, triggerfile = row['source_file_acquisition_full'], None
        print("Processing file {}".format(srcfile))
        num_data, data, targets, metadata = file_converter(srcfile, triggerfile)
        X.extend(data)
        Y.extend(targets)
        writer.writerows(metadata)
        print("Extracted: {} frames".format(num_data))
        num_frames += num_data
        print("Dataset current total frame count: {}".format(num_frames))

print('Creating dataset "{}" containing {} frames'.format(args.name, num_frames))

out_x = os.path.join(args.outdir, args.name + '_x')
out_y = os.path.join(args.outdir, args.name + '_y')
np.save(out_x, X)
np.save(out_y, Y)
