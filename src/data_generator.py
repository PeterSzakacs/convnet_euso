import sys
import math
import random as rand
import collections as coll

import numpy as np

import cmdint.cmd_interface_datagen as cmd
import utils.packets.packet_utils as pack
import utils.data_utils as dat

# declarations of custom functions (they use certain variables and lambdas declared later in this file)
def create_shower_packet(template, angle, shower_max, duration, max_EC_malfunctions=0):
    generator.reset(shower_max, duration)
    # let start coordinate be at least a distance of 3/4 * duration from the edges of the frame
    top, right, bottom, left = edge_generator(int(3*duration/4), template)

    start_gtu = 2
    start_x = rand.randrange(left, right)
    start_y = rand.randrange(bottom, top)
    start = (start_gtu, start_x, start_y)
    ang_rad = math.radians(angle)

    width, height, num_frames = template.frame_width, template.frame_height, template.num_frames
    packet = np.random.poisson(lam=lam, size=(num_frames, height, width))
    ECs_used = manipulator.draw_simulated_shower_line(packet, start, ang_rad, generator)

    frequencies = coll.Counter(ECs_used)
    shower_indexes = [item[0] for item in frequencies.most_common(2)]
    manipulator.simu_EC_malfunction(packet, max_EC_malfunctions, excluded_ECs=shower_indexes)
    return packet

def create_noise_packet(template, max_EC_malfunctions=0):
    width, height, num_frames = template.frame_width, template.frame_height, template.num_frames
    packet = np.random.poisson(lam=lam, size=(num_frames, height, width))
    manipulator.simu_EC_malfunction(packet, max_EC_malfunctions)
    return packet


# command line parsing

ui = cmd.cmd_interface()
args = ui.get_cmd_args(sys.argv[1:])
print(args)

# variable initialization

EC_width, EC_height = 16, 16
template = pack.packet_template(EC_width, EC_height, args.width, args.height, args.num_merged)
num_samples = args.num_frames
lam, bg_diff = args.lam, args.bg_diff

manipulator = dat.packet_manipulator(template, verify_against_template=False)
generator = dat.default_vals_generator(bg_diff, 10)

# use random shower duration
d_min, d_max = args.duration[:]
duration_generator = lambda: rand.randint(d_min, d_max)
# and calculate on every call limits for top, right, bottom and left margins of start coordinates
edge_generator = lambda edge_limit, template: (int(template.frame_height - edge_limit), int(template.frame_width - edge_limit), int(0 + edge_limit), int(0 + edge_limit))
if d_min == d_max:
    ## return static or precalculated values
    duration_generator = lambda: d_min
    top, right, bottom, left = edge_generator(3*duration_generator()/4, template)
    edge_generator = lambda template, edge_limit: (top, right, bottom, left)

# malfunctioned ECs simulation
EC_min, EC_max = args.bad_ECs[:]
maxerr_EC_generator = lambda: rand.randint(EC_min, EC_max)
if EC_min == EC_max:
    ## return static value
    maxerr_EC_generator = lambda: EC_min

# output and target generation
num_showers = int(num_samples/2)
iteration_handlers = (
    {'target': [1, 0], 'start': 0, 'stop': int(num_showers/2),
                        'packet_handler': lambda angle: create_shower_packet(template, angle, bg_diff, duration_generator(), maxerr_EC_generator())},
    {'target': [1, 0], 'start': int(num_showers/2), 'stop': num_showers,
                        'packet_handler': lambda angle: create_shower_packet(template, angle, bg_diff, duration_generator())},
    {'target': [0, 1], 'start': num_showers, 'stop': num_samples - int(num_showers/2),
                        'packet_handler': lambda angle: create_noise_packet(template, maxerr_EC_generator())},
    {'target': [0, 1], 'start': num_samples - int(num_showers/2), 'stop': num_samples,
                        'packet_handler': lambda angle: create_noise_packet(template)}
)
data = np.empty((num_samples, template.frame_height, template.frame_width), dtype=np.int32)
targets = np.empty((num_samples, 2), dtype=np.uint8)
# main loop
for handler in iteration_handlers:
    start, stop = handler['start'], handler['stop']
    packet_handler = handler['packet_handler']
    target = handler['target']
    # idx serves as both an index into targets and data, as well as shower angle in xy projection
    for idx in range(start, stop):
        packet = packet_handler(idx)
        data[idx] = manipulator.create_x_y_projection(packet)
        np.put(targets[idx], [0, 1], target)


# shuffle the data in unison for a given number of times
for idx in range(args.num_shuffles):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

# save generated data and targets
np.save(args.outfile, data)
np.save(args.targetfile, targets)
