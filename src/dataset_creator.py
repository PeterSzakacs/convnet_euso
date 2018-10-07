import sys
import csv

import utils.dataset_utils as ds
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
    global data, targets, metadata_dict, helper, num_data_counter
    result = helper.convert_packet(packet, start_idx=27, end_idx=47)
    for idx in range(len(data)):
        data[idx].append(result[idx])
    metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
    targets.append([0, 1])
    num_data_counter += 1

def on_packet_extracted_simu(packet, packet_idx, srcfile):
    if (packet_idx == 1):
        global data, targets, metadata_dict, helper, num_data_counter
        data_len = len(data)
        result = helper.convert_packet(packet, start_idx=27, end_idx=47)
        for idx in range(data_len):
            data[idx].append(result[idx])
        result = helper.convert_packet(packet, start_idx=0, end_idx=20)
        metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
        targets.append([1, 0])
        for idx in range(data_len):
            data[idx].append(result[idx])
        metadata_dict.append({'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx})
        targets.append([0, 1])
        num_data_counter += 2

# command line parsing

cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])
print(args)

# variable initialization (might need to transform these into user-settable cmd args)

EC_width, EC_height = 16, 16
frame_width, frame_height = 48, 48
num_frames_per_packet = 128
packet_template = pack.packet_template(EC_width, EC_height, frame_width, frame_height, num_frames_per_packet)
extractor = iout.packet_extractor(packet_template=packet_template)

# globals
helper = args.helper
## We do not know how many data items there will be, as the input tsv might not have information
## on how many packets there are per file
data = helper.create_converted_packets_holders(1, (1, 1, 1))
data = tuple([] for holder in data)
targets, metadata_dict = [], []
num_data_counter = 0

if args.simu:
    data_extractor = lambda srcfile, triggerfile: extractor.extract_packets_from_npyfile_and_process(
                                                        srcfile, triggerfile=triggerfile,
                                                        on_packet_extracted=on_packet_extracted_simu)
else:
    data_extractor = lambda srcfile, triggerfile: extractor.extract_packets_from_rootfile_and_process(
                                                        srcfile, triggerfile=triggerfile,
                                                        on_packet_extracted=on_packet_extracted_flight)

input_tsv = args.filelist
output_tsv = args.metafile
total_num_data = 0

# We do not know how many data items there will be, as the input tsv might not have information
# on how many packets there are per file
dataset_holders = helper.create_converted_packets_holders(1, (1, 1, 1))
dataset_holders = tuple([] for holder in dataset_holders)

# main loop
with open(input_tsv, 'r') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    for row in reader:
        srcfile, triggerfile = row['source_file_acquisition_full'], None
        print("Processing file {}".format(srcfile))
        num_data_counter = 0
        data_extractor(srcfile, triggerfile)
        total_num_data += num_data_counter
        print("Extracted: {} data items".format(num_data_counter))
        print("Dataset current total data items count: {}".format(total_num_data))


print('Creating dataset "{}" containing {} items'.format(args.name, total_num_data))

# write out dataset metadata
with open(output_tsv, 'w') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=['source_file_acquisition_full', 'packet_idx'], delimiter='\t')
    writer.writeheader()
    writer.writerows(metadata_dict)

# save dataset
ds.save_dataset(data, targets, args.outfiles, args.metafile)
