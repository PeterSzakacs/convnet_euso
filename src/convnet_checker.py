# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import math
import csv
import os
import importlib
import datetime as dt

import tflearn
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

import visualization.html_writers as html
import utils.dataset_utils as ds
import utils.network_utils as netutils

def save_frame(frame, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(frame)
    # A few input ROOT files cause this line to crash the program, will have to check later
    #plt.colorbar(im)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':

    import sys
    import cmdint.cmd_interface_checker as cmd

    # command line argument parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    # do not use the GPU
    if args.usecpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # load input data, targets and metadata
    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name)

    # deprecated feature, as this was never really used.
    # else:
        # args.infile = args.acqfile
        # args.targetfile = 'None (implicitly assumed_noise in all packets)'
        # extractor = io_utils.packet_extractor()
        # manipulator = dat.packet_manipulator(extractor.packet_template)
        # X_all = []
        # proj_creator = lambda packet, packet_idx, srcfile: X_all.append(
        #                             manipulator.create_x_y_projection(packet, start_idx=27, end_idx=47))
        # extractor.extract_packets_from_rootfile_and_process(args.acqfile, triggerfile=args.triggerfile, on_packet_extracted=proj_creator)
        # # implicitly assuming that all packets will contain noise
        # X_all = np.array(X_all, dtype=np.uint8)
        # Y_all = np.array([[0, 1] for idx in range(len(X_all))], dtype=np.uint8)

    # load metadata
    # metadata = []
    # if args.metafile != None:
    #     # go to next csv row and get 'source_file_acquisition_full'
    #     with open(args.metafile) as metafile:
    #         reader = csv.DictReader(metafile, delimiter='\t')
    #         for row in reader:
    #             metadata.append((row['source_file_acquisition_full'], row['packet_idx']))
    #     args.metafile = os.path.abspath(args.metafile)
    # else:
    #     # just write infile as source file of packet
    #     metadata = [(args.name, idx) for idx in range(num_data)]

    # run_time = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # # prepare logging directory
    # if not os.path.exists(args.logdir):
    #     os.mkdir(args.logdir)
    # ## new subdirectory for logging the results of the current script invocation
    # current_run_dir = os.path.join(args.logdir, run_time)
    # os.mkdir(current_run_dir)

    # # frame creation callback
    # if args.noframes:
    #     frame_creator = lambda frame, figurename, outdir: None
    # else:
    #     frame_creator = lambda frame, figurename, outdir: save_frame(frame, os.path.join(outdir, figurename))
    # frame_saver = lambda frames, figurenames, outdir: [frame_creator(frames[idx], figurenames[idx], outdir) for idx in range(inputs_len)]

    # # set maximum table size
    # table_size = args.tablesize or 2500
    # table_size = 2500 if table_size > 2500 else table_size

    # # html writers
    # fil = html.file_writer()
    # tbl = html.table_writer()
    # img = html.image_writer()
    # txt = html.text_writer()

    # main loop
    for network in args.networks:
        network_module_name, model_file = network[0], network[1]
        network_module_name = "net." + network_module_name
        run_id = netutils.get_default_run_id(network_module_name);
        tb_dir = os.path.join(netutils.DEFAULT_CHECKING_LOGDIR, network_module_name)
        model, net, conv, fc = netutils.import_convnet(
            network_module_name, tb_dir, dataset.item_shapes, model_file=model_file
        )
        #logdir = os.mkdir(os.path.join(current_run_dir, network_name))
        log_data, hits, classes_count = netutils.evaluate_classification_model(
            model, dataset, slice(args.eval_numframes))
        # sort ascending in the direction of greater noise probability
        log_data.sort(key=lambda item: item[1][1])

        # Misc statistics
        shower_count, noise_count = classes_count[0], classes_count[1]
        acc = (hits * 100 / dataset.num_data)
        err = 100 - acc

        print('acc: {}, err: {}, showers: {}, noise: {}'.format(acc, err, shower_count, noise_count))

        # # for every 2500 records of logs create a separate html report
        # globalstop = len(log_data)
        # iteration, last_iteration = 0, math.floor(globalstop / table_size)
        # for firstrow_idx in range(0, globalstop, table_size):
        #     fil.begin_html_file(title="Report for network {}".format(network_name),
        #                         css_rules="""
        #                                     table { border: 1px solid black; }
        #                                     .l { float: left; }
        #                                     .r { float: right; }
        #                                     .parspan { width: 100%; min-height: 20px}
        #                                 """)
        #     txt.begin_list()
        #     txt.add_heading("Statistics", level=2)
        #     txt.add_list_item("Time run: {}".format(run_time))
        #     #txt.add_list_item("Dataset file used for test: {}".format(args.infile))
        #     #txt.add_list_item("Targets of dataset file: {}".format(args.targetfile))
        #     #txt.add_list_item("Optional metadata file: {}".format(args.metafile))
        #     txt.add_list_item("Network architecture name: {}".format(network_name))
        #     txt.add_list_item("Trained model file: {}".format(os.path.abspath(model_file)))
        #     txt.add_list_item("Number of items predicted as shower: {}".format(shower_count))
        #     txt.add_list_item("Number of items predicted as noise: {}".format(noise_count))
        #     txt.add_list_item("Total number of items checked: {}".format(num_data))
        #     txt.add_list_item("Accuracy: {}%".format(round(acc, 2)))
        #     txt.add_list_item("Error rate: {}%".format(round(err, 2)))
        #     txt.end_list()
        #     txt.add_heading("Table of frames", level=2)
        #     prevfile = "#" if iteration == 0 else "report_{}.html".format(iteration - 1)
        #     nextfile = "#" if iteration == last_iteration else "report_{}.html".format(iteration + 1)
        #     l = txt.get_link(href=prevfile, link_contents="prev", styleclass="l")
        #     r = txt.get_link(href=nextfile, link_contents="next", styleclass="r")
        #     fil.add_div(l + r, styleclass="parspan")
        #     tbl.begin_table(table_headings=(*["Frame yx", "Frame gtux", "Frame gtuy", "Shower %", "Noise %", "Output", "Target", "Source file", "Index of packet"))
        #     localstop = firstrow_idx + table_size if firstrow_idx + table_size < globalstop else globalstop
        #     # Build html table
        #     for log_idx in range(firstrow_idx, localstop, 1):
        #         log = log_data[log_idx]
        #         frame_idx, prediction = log[0], log[1]
        #         srcfile, packet_idx = metadata[frame_idx][0], metadata[frame_idx][1]
        #         rounded_prediction = np.round(prediction).astype(np.uint8)
        #         shower_prob = round(prediction[0] * 100, 2)
        #         noise_prob = round(prediction[1] * 100, 2)
        #         out = 'noise' if np.array_equal(rounded_prediction, [0, 1]) else 'shower'
        #         targ = 'noise' if np.array_equal(targets[frame_idx], [0, 1]) else 'shower'
        #         figurenames = ['frame-{}-{}.svg'.format(frame_idx, itype)
        #                        for itype in input_types
        #                        if input_shapes[itype] != None]
        #         frame_saver(item_getter(frame_idx), figurenames, logdir)
        #         tbl.append_table_row(row_contents=(img.get_image(figurename + '.svg', width="184px", height="138px"),
        #                                         "{}%".format(shower_prob), "{}%".format(noise_prob), out, targ, srcfile, packet_idx))
        #     tbl.end_table()
        #     fil.end_html_file(os.path.join(logdir, 'report_{}.html'.format(iteration)))
        #     iteration += 1
