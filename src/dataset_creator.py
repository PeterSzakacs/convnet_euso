import os
import sys
import csv

import numpy as np

import utils.data_utils as dat
import utils.io_utils as iout
import utils.packets.packet_utils as pack
import cmdint.cmd_interface_creator as cmd



# script to coallesce (simulated or real) data from several files 
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored

# declarations of custom functions (they use certain variables declared later in this file)
# for creating projections from the extracted packets, targets and metadata for them
def on_packet_extracted_flight(packet, packet_idx, srcfile):
    global data, targets, metadata_dict
    data[0].append(manipulator.create_x_y_projection(packet, start_idx=27, end_idx=47))
    data[1].append(manipulator.create_x_gtu_projection(packet, start_idx=27, end_idx=47))
    data[2].append(manipulator.create_y_gtu_projection(packet, start_idx=27, end_idx=47))
    metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
    targets.append([0, 1])

def on_packet_extracted_simu(packet, packet_idx, srcfile):
    if (packet_idx == 1):
        global data, targets, metadata_dict
        data[0].append(manipulator.create_x_y_projection(packet, start_idx=27, end_idx=47))
        data[1].append(manipulator.create_x_gtu_projection(packet, start_idx=27, end_idx=47))
        data[2].append(manipulator.create_y_gtu_projection(packet, start_idx=27, end_idx=47))
        metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
        targets.append([1, 0])
        data[0].append(manipulator.create_x_y_projection(packet, start_idx=0, end_idx=20))
        data[1].append(manipulator.create_x_gtu_projection(packet, start_idx=0, end_idx=20))
        data[2].append(manipulator.create_y_gtu_projection(packet, start_idx=0, end_idx=20))
        metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
        targets.append([0, 1])

# command line parsing

cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])

# variable initialization (might need to transform these into user-settable cmd args)

EC_width, EC_height = 16, 16
frame_width, frame_height = 48, 48
num_frames_per_packet = 128

data, targets, metadata_dict = ([], [], []), [], []
packet_template = pack.packet_template(EC_width, EC_height, frame_width, frame_height, num_frames_per_packet)
manipulator = dat.packet_manipulator(packet_template, verify_against_template=True)
extractor = iout.packet_extractor(packet_template=packet_template)

if args.simu:
    data_extractor = lambda srcfile, triggerfile: extractor.extract_packets_from_npyfile_and_process(
                                                        srcfile, triggerfile=triggerfile, 
                                                        on_packet_extracted=on_packet_extracted_simu)
else:
    data_extractor = lambda srcfile, triggerfile: extractor.extract_packets_from_rootfile_and_process(
                                                        srcfile, triggerfile=triggerfile, 
                                                        on_packet_extracted=on_packet_extracted_flight)

input_tsv = args.filelist
output_tsv = os.path.join(args.outdir, args.name + '_meta.tsv')
total_num_data = 0
YX_proj, GTU_X_proj, GTU_Y_proj, Y = [], [], [], []
# main loop
with open(input_tsv, 'r') as infile, open(output_tsv, 'w') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    writer = csv.DictWriter(outfile, fieldnames=['source_file_acquisition_full', 'packet_idx'], delimiter='\t')
    writer.writeheader()
    for row in reader:
        srcfile, triggerfile = row['source_file_acquisition_full'], None
        print("Processing file {}".format(srcfile))
        data_extractor(srcfile, triggerfile)
        YX_proj.extend(data[0])
        GTU_X_proj.extend(data[1])
        GTU_Y_proj.extend(data[2])
        Y.extend(targets)
        num_data = len(data[0])
        total_num_data += num_data
        print("Extracted: {} data items".format(num_data))
        print("Dataset current total data items count: {}".format(total_num_data))
        data[0].clear()
        data[1].clear()
        data[2].clear()
        targets.clear()
    # write file metadata out all at once
    writer.writerows(metadata_dict)

print('Creating dataset "{}" containing {} items'.format(args.name, total_num_data))

# save dataset

out_y_x = os.path.join(args.outdir, args.name + '_yx')
out_gtu_x = os.path.join(args.outdir, args.name + '_gtux')
out_gtu_y = os.path.join(args.outdir, args.name + '_gtuy')
np.save(out_y_x, YX_proj)
np.save(out_gtu_x, GTU_X_proj)
np.save(out_gtu_y, GTU_Y_proj)

out_y = os.path.join(args.outdir, args.name + '_targets')
np.save(out_y, Y)
