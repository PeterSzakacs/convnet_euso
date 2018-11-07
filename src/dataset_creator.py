import sys
import csv

import utils.dataset_utils as ds
import utils.io_utils as iout
import cmdint.cmd_interface_creator as cmd



# script to coallesce (simulated or real) data from several files
# into a single larger .npy file to use as a dataset for training
# or evaluating a network.

# currently no usage for L1 trigger file, so it is just ignored

# declarations of custom functions (they use certain variables declared later in this file)
# for creating projections from the extracted packets, targets and metadata for them
def on_packet_extracted_flight(packet, packet_idx, srcfile):
    global dataset, num_data_counter
    metadata_dict = {'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx}
    dataset.add_data_item(packet[27:47], [0, 1], metadata_dict)
    num_data_counter += 1

def on_packet_extracted_simu(packet, packet_idx, srcfile):
    if (packet_idx == 1):
        global dataset, num_data_counter
        metadata_dict = {'source_file_acquisition_full': srcfile, 'packet_idx': packet_idx}
        dataset.add_data_item(packet[27:47], [1, 0], metadata_dict)
        dataset.add_data_item(packet[0:20], [0, 1], metadata_dict)
        num_data_counter += 2

# command line parsing

cmd_int = cmd.cmd_interface()
args = cmd_int.get_cmd_args(sys.argv[1:])
print(args)

# variable initialization (might need to transform these into user-settable cmd args)

packet_template = args.template
extractor = iout.packet_extractor(packet_template=packet_template)
# the data items are only ever made from 20 frames from the extracted packet
extracted_packet_shape = list(packet_template.packet_shape)
extracted_packet_shape[0] = 20

# globals
## We do not know how many data items there will be, as the input tsv might not have information
## on how many packets there are per file
dataset = ds.numpy_dataset(args.name, extracted_packet_shape, item_types=args.item_types)
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
total_num_data = 0

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
dataset.save(args.outdir)
