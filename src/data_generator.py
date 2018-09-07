import sys
import math
import numpy as np
import random as rand
import collections as coll

import cmdint.cmd_interface_datagen as cmd
import utils.packet_tools as pack

# command line parsing

ui = cmd.cmd_interface()
args = ui.get_cmd_args(sys.argv[1:])
print(args)

# variable initialization

width, height = args.width, args.height
EC_width, EC_height = 16, 16
num_per_projection = args.num_merged
num_samples = args.num_frames
lam, bg_diff = args.lam, args.bg_diff

# declarations of custom functions and value generators

manipulator = pack.packet_manipulator(EC_width, EC_height, width, height)
generator = pack.flat_vals_generator(bg_diff, 10)
## use random shower line lengths, unless duration of shower is explicitly stated
duration_generator = lambda: rand.randrange(3, 16)
## return limits for top, right, bottom and left start margins
offsets_generator = lambda edge_limit: (int(height - edge_limit), int(width - edge_limit), int(0 + edge_limit), int(0 + edge_limit))
if args.duration != None:
    ### return static value
    duration_generator = lambda: args.duration
    ### return precalculated values
    top, right, bottom, left = offsets_generator(3*duration_generator()/4)
    offsets_generator = lambda edge_limit: (top, right, bottom, left)

def create_shower_packet(angle, shower_max, max_EC_malfunctions):
    duration = duration_generator()
    generator.reset(shower_max, duration)
    top, right, bottom, left = offsets_generator(3*duration/4)
    start_x = rand.randrange(left, right)
    start_y = rand.randrange(bottom, top)
    ang_rad = math.radians(angle)

    packet = np.random.poisson(lam=lam, size=(num_per_projection, width, height))
    ECs_used = manipulator.draw_simulated_shower_line(packet, start_x, start_y, ang_rad, generator)
    frequencies = coll.Counter(ECs_used)
    shower_indexes = [item[0] for item in frequencies.most_common(2)]
    manipulator.simu_EC_malfunction(packet, max_EC_malfunctions, shower_EC_indexes=shower_indexes)
    return packet

def create_noise_packet(max_EC_malfunctions):
    packet = np.random.poisson(lam=lam, size=(num_per_projection, width, height))
    manipulator.simu_EC_malfunction(packet, max_EC_malfunctions)
    return packet

# by default, do not simulate malfunctioned EC cells
num_showers = int(num_samples/2)
iteration_handlers = (
    {'target': [1, 0], 'start': 0, 'stop': num_showers, 'track_ECs': False,
                            'packet_handler': lambda angle: create_shower_packet(angle, bg_diff, 0)},
    {'target': [0, 1], 'start': num_showers, 'stop': num_samples, 'track_ECs': False,
                            'packet_handler': lambda angle: create_noise_packet(0)}
)
if args.malfunctioning_EC:
    maxerr_EC_generator = lambda: rand.randrange(1, int(manipulator.num_EC - 1))
    iteration_handlers = (
        {'target': [1, 0], 'start': 0, 'stop': int(num_showers/2), 'track_ECs': True,
                            'packet_handler': lambda angle: create_shower_packet(angle, bg_diff, maxerr_EC_generator())},
        {'target': [1, 0], 'start': int(num_showers/2), 'stop': num_showers, 'track_ECs': False,
                            'packet_handler': lambda angle: create_shower_packet(angle, bg_diff, 0)},
        {'target': [0, 1], 'start': num_showers, 'stop': num_samples - int(num_showers/2), 'track_ECs': True,
                            'packet_handler': lambda angle: create_noise_packet(maxerr_EC_generator())},
        {'target': [0, 1], 'start': num_samples - int(num_showers/2), 'stop': num_samples, 'track_ECs': False,
                            'packet_handler': lambda angle: create_noise_packet(0)}
    )

# output and target generation
data = np.empty((num_samples, width, height), dtype=np.int32)
targets = np.empty((num_samples, 2), dtype=np.uint8)
# main loop
for handler in iteration_handlers:
    start, stop = handler['start'], handler['stop']
    packet_handler = handler['packet_handler']
    target = handler['target']
    manipulator.track_ECs = handler['track_ECs']
    # idx serves as both an index into targets and data, as well as shower angle in xy projection
    for idx in range(start, stop):
        packet = packet_handler(idx)
        data[idx] = np.max(packet, axis=0)
        np.put(targets[idx], [0, 1], target)


# shuffle the data in unison 7 times
for idx in range(args.num_shuffles):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

# save generated data and targets
np.save(args.outfile, data)
np.save(args.targetfile, targets)
