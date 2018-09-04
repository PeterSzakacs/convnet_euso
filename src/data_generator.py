import sys
import math
import numpy as np
import random as rand
from collections import Counter

import cmdint.cmd_interface_datagen as cmd

# generator for shower line values
class shower_vals_generator():
    
    def __init__(self, maximum, duration):
        self.reset(maximum, duration)
        
    def reset(self, maximum, duration):
        self.iteration = -3*duration/4
        self.maximum = maximum
        self.maxinv = 1/maximum

    def next_val(self):
        self.iteration += 1
        return -self.maxinv * pow(self.iteration - 2, 2) + self.maximum

# command line parsing

ui = cmd.cmd_interface()
args = ui.get_cmd_args(sys.argv[1:])

# variable initialization

num_samples = args.num_frames
num_showers = int(num_samples/2)
width, height = args.width, args.height
lam, bg_diff = args.lam, args.bg_diff

# use random shower line lengths, unless duration of shower is explicitly stated
duration_generator = lambda: rand.randrange(3, 16)
if args.duration != None:
    # return static value
    duration_generator = lambda: args.duration

# return limits for top, right, bottom and left start margins
offsets_generator = lambda edge_limit: (height - edge_limit, width - edge_limit, 0 + edge_limit, 0 + edge_limit)
if args.duration != None:
    # return precalculated values
    top, right, bottom, left = offsets_generator(duration_generator())
    offsets_generator = lambda edge_limit: (top, right, bottom, left)

num_per_frame = args.num_merged
num_rows = height / 16
num_regions = width * height / 16*16


# output and target generation

# First loop: generate frames containing a simulated air shower
data = np.empty((num_samples, width, height), dtype=np.int32)
targets = np.empty((num_samples, 2), dtype=np.uint8)
generator = shower_vals_generator(bg_diff, 10)
for angle in range(num_showers):
    angRad = math.radians(angle)
    duration = duration_generator()
    generator.reset(bg_diff, duration)
    top, right, bottom, left = offsets_generator(duration)
    startX = rand.randrange(left, right)
    startY = rand.randrange(bottom, top)
    angRad = math.radians(angle)
    deltaX = math.sin(angRad)
    deltaY = math.cos(angRad)
    # Create background for the frame
    all_frames = np.random.poisson(lam=lam, size=(num_per_frame, width, height))
    for idx in range(2, 2 + duration, 1):
        frame = all_frames[idx]
        offsetX = startX + math.floor(deltaX * idx)
        offsetY = startY + math.floor(deltaY * idx)
        nextval = max(round(generator.next_val()), 0)
        if offsetX < 0 or offsetX >= 48 or offsetY < 0 or offsetY > 48:
            break
        frame[offsetX][offsetY] += nextval

    # Target (expected ouptut) for shower frame is [1, 0]
    np.put(targets[angle], [0, 1], [1, 0])

    data[angle] = np.max(all_frames, axis=0)


# Second loop: generate random noise without showers
for idx in range(num_showers, num_samples):
    all_frames = np.random.poisson(lam=lam, size=(num_per_frame, width, height))
    data[idx] = np.max(all_frames, axis=0)
    # Target (expected ouptut) for pure noise frame is [0, 1]
    np.put(targets[idx], [0, 1], [0, 1])


# shuffle the data in unison
rng_state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(rng_state)
np.random.shuffle(targets)

# save generated data and targets
np.save(args.outfile, data)
np.save(args.targetfile, targets)
