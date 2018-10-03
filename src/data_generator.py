import sys
import dis
import math
import operator
import random as rand

import numpy as np

import utils.packets.packet_utils as pack
import utils.data_utils as dat

def create_shower_packet(props_dict, angle, max_EC_malfunctions=0):
    # get a start coordinate and calculate parameters to pass to the method of packet_manipulator
    start_gtu = props_dict['start_gtu_generator'](*props_dict['start_gtu'])
    start_x = props_dict['start_x_generator'](*props_dict['start_x']) 
    start_y = props_dict['start_y_generator'](*props_dict['start_y'])
    start = (start_gtu, start_x, start_y)
    ang_rad = math.radians(angle)
    shower_max = props_dict['bg_diff_generator'](*props_dict['bg_diff'])
    duration = props_dict['duration_generator'](*props_dict['duration'])
    values_generator = props_dict['vals_generator']
    values_generator.reset(shower_max, duration)

    # create the actual packet
    manipulator = props_dict['manipulator']
    template, lam = props_dict['template'], props_dict['bg_lambda']
    packet = np.random.poisson(lam=lam, size=template.packet_shape)
    X, Y, GTU, vals = manipulator.draw_simulated_shower_line(start, ang_rad, values_generator)
    packet[GTU, Y, X] += vals

    # get the sum of shower pixel values in all EC modules
    ECs_used = [template.xy_to_ec_idx(x, y) for (x, y) in zip(X, Y)]
    ECs_dict = dict(zip(ECs_used, [0]*len(ECs_used)))
    for idx in range(len(ECs_used)):
        EC = ECs_used[idx]
        ECs_dict[EC] += packet[GTU[idx], Y[idx], X[idx]]
    # get the EC containing the maximum sum of pixel values
    maxval_EC = max(ECs_dict.items(), key=operator.itemgetter(1))[0]
    # zero-out pixels to simulate random EC failures
    X, Y, indices = manipulator.select_random_ECs(max_EC_malfunctions, excluded_ECs=[maxval_EC])
    for idx in range(len(indices)):
        packet[:, Y[idx], X[idx]] = 0
    return packet

def create_noise_packet(props_dict, max_EC_malfunctions=0):
    manipulator = props_dict['manipulator']
    template, lam = props_dict['template'], props_dict['bg_lambda']
    packet = np.random.poisson(lam=lam, size=template.packet_shape)
    X, Y, indices = manipulator.select_random_ECs(max_EC_malfunctions)
    for idx in range(len(indices)):
        packet[:, Y[idx], X[idx]] = 0
    return packet

"""Generates and returns a set of data containing simulated shower lines and corresponding targets,
both as numpy arrays, for use in training neural networks for classifiction tasks.

Individual data items are created by first creating a packet for each data item and then creating 
projections which are the actual data items returned along with targets for them.

The data returned is divided into quarters of equal size containing the following data:
1/4: shower data (possibly with malfunctioned EC units)
2/4: shower data (without malfunctioned EC units)
3/4: noise data (possibly with malfunctioned EC units)
4/4: noise data (without malfunctioned EC units)

Whether there are any data items with malfunctioning ECs depends on the value of the parameter 
packet_props_dict['bad_ECs']. By default no attempt is made to simulate their effect.

Parameters
----------
num_data :          int
                    The number of data items to create in total
packet_props_dict : dict of str: any
                    Dictionary of properties used in packet creation (lambda value for Poisson distributions, packet template, 
                    number of bad EC units etc.). 
shower_props_dict : dict of str: tuple of int
                    Dictionary of value ranges (represented as tuples of 2 ints) for specific shower properties (start coordinates, 
                    duration, background difference)
Returns
-------
data :      tuple of np.ndarray
            A tuple where each item is a numpy array containing packet projections of a given type (xy, xgtu, ygtu) numbering 
            num_data projections for each type
targets :   np.ndarray
            Numpy array containing classification targets for each data item at the same index (for any projection type)
"""
def create_dataset(num_data, packet_props_dict, shower_props_dict):
    # TODO: might want to break this up into even smaller functions for better testing
    template = packet_props_dict['template']
    # properties dictionary initialization
    props = dict()
    props['template'], props['bg_lambda'] = template, packet_props['bg_lambda']
    for shower_param in shower_props_dict:
        min, max = shower_props_dict[shower_param][0:2]
        key = shower_param + '_generator'
        props[key] = (lambda min, max: min) if min == max else (lambda min, max: rand.randint(min, max))
        props[shower_param] = (min, max)
    manipulator = dat.packet_manipulator(template, verify_against_template=False)
    props['manipulator'] = manipulator
    props['vals_generator'] = dat.default_vals_generator(1, 10)

    # malfunctioned ECs simulation
    EC_min, EC_max = packet_props_dict['bad_ECs'][:]
    maxerr_EC_generator = (lambda min, max: min) if EC_min == EC_max else (lambda min, max: rand.randint(min, max))

    # output and target generation
    num_samples = num_data
    num_showers = int(num_samples/2)
    iteration_handlers = (
        {'target': [1, 0], 'start': 0, 'stop': int(num_showers/2),
                            'packet_handler': lambda angle: create_shower_packet(props, angle, maxerr_EC_generator(EC_min, EC_max))},
        {'target': [1, 0], 'start': int(num_showers/2), 'stop': num_showers,
                            'packet_handler': lambda angle: create_shower_packet(props, angle)},
        {'target': [0, 1], 'start': num_showers, 'stop': num_samples - int(num_showers/2),
                            'packet_handler': lambda angle: create_noise_packet(props, maxerr_EC_generator(EC_min, EC_max))},
        {'target': [0, 1], 'start': num_samples - int(num_showers/2), 'stop': num_samples,
                            'packet_handler': lambda angle: create_noise_packet(props)}
    )
    yx_proj = np.empty((num_samples, template.frame_height, template.frame_width), dtype=np.int32)
    gtux_proj = np.empty((num_samples, template.num_frames, template.frame_width), dtype=np.int32)
    gtuy_proj = np.empty((num_samples, template.num_frames, template.frame_height), dtype=np.int32)
    data = (yx_proj, gtux_proj, gtuy_proj)
    targets = np.empty((num_samples, 2), dtype=np.uint8)
    # main loop
    for handler in iteration_handlers:
        start, stop = handler['start'], handler['stop']
        packet_handler = handler['packet_handler']
        target = handler['target']
        # idx serves as both an index into targets and data, as well as shower angle in xy projection
        for idx in range(start, stop):
            packet = packet_handler(idx)
            data[0][idx] = manipulator.create_x_y_projection(packet)
            data[1][idx] = manipulator.create_x_gtu_projection(packet)
            data[2][idx] = manipulator.create_y_gtu_projection(packet)
            np.put(targets[idx], [0, 1], target)
    
    return data, targets

"""Shuffle generated data and their targets in unison for a given number of times.

Parameters
----------
num_shuffles :  int
                number of times the data and targets are to be shuffled
dataset :       tuple of numpy.ndarray
                the dataset in the form of a tuple of numpy arrays (the result of calling create_dataset())
targets :       numpy.ndarray
                the expected classification outputs (targets) for a neural network (the result of calling create_dataset())
"""
def shuffle_dataset(num_shuffles, dataset, targets):
    for idx in range(num_shuffles):
        rng_state = np.random.get_state()
        for idx in range(len(dataset)):
            np.random.shuffle(dataset[idx])
            np.random.set_state(rng_state)
        np.random.shuffle(targets)

# save generated data and targets
def save_dataset(dataset, targets, dataset_filenames, targets_filename):
    for idx in range(len(dataset)):
        np.save(dataset_filenames[idx], dataset[idx])
    np.save(targets_filename, targets)



if __name__ == '__main__':
    import cmdint.cmd_interface_datagen as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    num_data = args.num_data
    shower_props = args.shower_properties
    packet_props = {'bg_lambda': args.bg_lambda, 'template': args.template, 'bad_ECs': args.bad_ECs}
    data, targets = create_dataset(num_data, packet_props, shower_props)
    
    shuffle_dataset(args.num_shuffles, data, targets)
    save_dataset(data, targets, args.outfiles, args.targetfile)
