import sys
import math
import utils.cmd_interface_datagen as cmd
import numpy as np
import random as rand
from skimage.draw import line_aa

ui = cmd.cmd_interface()
args = ui.get_cmd_args(sys.argv[1:])

# variable initialization

width, height = args.width, args.height
lam, bgDiff = args.lam, args.bg_diff
num_samples = args.num_frames
num_showers = int(num_samples/2)

# output and target generation

data = np.random.poisson(lam=lam, size=(num_samples, width, height))
targets = np.empty((num_samples, 2), dtype=np.uint8).astype(np.uint8)
for idx in range(num_showers):
    angRad = math.radians(idx)
    startX = rand.randrange(3, width - 3)
    startY = rand.randrange(3, height - 3)
    offsetX = round(10*math.sin(angRad))
    offsetY = round(10*math.cos(angRad))
    endX = 0; endY = 0;
    if (offsetX < 0):
        endX = np.minimum(startX + offsetX, 0)
    else:
        endX = np.minimum(startX + offsetX, width - 1)
    if (offsetY < 0):
        endX = np.minimum(startY + offsetY, 0)
    else:
        endX = np.minimum(startY + offsetY, height - 1)
    rr, cc, val = line_aa(startX, startY, endX, endY)
    b = data[idx]
    b[rr, cc] += (val * bgDiff).astype(np.int32)
    np.put(targets[idx], [0, 1], [1, 0])

for idx in range(num_showers, num_samples):
    np.put(targets[idx], [0, 1], [0, 1])


# shuffle the data in unison

rng_state = np.random.get_state()
np.random.shuffle(data)
np.random.set_state(rng_state)
np.random.shuffle(targets)

# save generated data and targets

np.save(args.outfile, data)
np.save(args.targetfile, targets)
