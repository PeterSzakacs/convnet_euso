""" Script to generate simulated air shower data along with expected 
    neural network classification outputs (targets).

    The output of this script is a pair of .npy files: 
    - the data (x) with dimensions (num_frames, width, height)
    - the targets (y) with dimensions (num_frames, 2)

    For a quick overview of how to use this script, simply 
    call it from the command line with option -h, e.g. 

    ~$ python data_generator2.py -h



    == Implementation details (verbose) ==

    Currently this script creates data simply by drawing an antialiased 
    line onto a background (bg) of random pixel values. A detailed explanation 
    is given below, but to aid in understanding "why" it is the way it is 
    you can check these example algorithms:

    https://stackoverflow.com/a/31671355
    http://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.line_aa

    The following are key parameters that directly influence the generated data
    (all of them are integers):

    width            - width in pixels of an individual frame
    height           - height in pixels of an individual frame
    num_frames       - total number of frames in the generated data, essentially 
                       the total number of items in the dataset (this means both 
                       showers and pure noise)
    lambda           - average bg noise value
    bg_diff          - maximum positive difference between the value of a bg pixel 
                       after the shower line is drawn onto it and before
    num_merged       - number of frames to merge for creating the actual bg
                       for a data item

    Currently all the data is generated with a 50:50 ratio between shower and 
    noise frames. For example, if num_frames is 20000, then this script will 
    generate 10000 frames containing a line and 10000 frames containing noise.
    
    Bg_diff refers to how much higher (at most) the pixel values of the line
    of the shower are in comparison to the background. To illustrate, suppose 
    the following are some example pixel values before the line is drawn:
    [0, 1, 3, 0, 1, 3, 1]
    then a line drawn passing through them will increment these by at most 
    bg_diff, for example for bg_diff == 5 it could look like this:
    [5, 6, 7, 4, 4, 7, 3]
    +5 +5 +4 +4 +3 +3 +2

    The reason this increment in value is "at most" bg_diff is because we are 
    drawing an antialiased line, meaning it can have brighter and darker spots 
    along its length. This is also the reason why the line is created by adding 
    values to relevant pixels, as opposed to simply replacing them, because 
    the line would then have spots significantly less bright than even the bg 
    surrounding it.

    Lambda refers to the average value of background pixels. This directly 
    corresponds to the lambda parameter found in Poisson distributions.

    Merge_num_frames represents a minor trick to make the bg look more like 
    what one would expect by merging several consecutive frames from a packet, 
    which tends to shift the distribution of bg pixel values away from Poisson.
    Essentially, we first create a number of frames with dimensions (width, height) 
    with a Poisson distribution using the value of lambda given and then merge them 
    into a single frame of size (width, height) using np.max(). It is not a perfect 
    imitation, but then again, neither is using an antialiased line. ¯\_(ツ)_/¯
    
"""

import sys
import math
import numpy as np
import random as rand
from skimage.draw import line_aa

import utils.cmdint.cmd_interface_datagen as cmd

# command line parsing

ui = cmd.cmd_interface()
args = ui.get_cmd_args(sys.argv[1:])

# variable initialization

width, height = args.width, args.height
lam, bgDiff = args.lam, args.bg_diff
num_per_frame = args.num_merged
num_samples = args.num_frames
num_showers = int(num_samples/2)

# output and target generation

# First loop: generate frames containing a simulated air shower
data = np.empty((num_samples, width, height), dtype=np.int32)
targets = np.empty((num_samples, 2), dtype=np.uint8)
for idx in range(num_showers):
    # Create background for the frame
    all_frames = np.random.poisson(lam=lam, size=(num_per_frame, width, height))
    data[idx] = np.max(all_frames, axis=0)

    # Pick a random starting coordinate and calculate the ending coordinate 
    # for a line segment of length 10 (this latter part does not work, 
    # probably because of differences in coordinate systems between skimage 
    # and numpy, though it seems to be for the better),
    ongRad = math.radians(idx)
    startX = rand.randrange(3, width - 3)
    startY = rand.randrange(3, height - 3)
    angRad = math.radians(idx)
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

    # Return relative pixel values (val, float from 0 to 1) 
    # for an antialiased line from (startX, startY) to (endX, endY) 
    # including X (rr) and Y (cc) coordinates for those values
    rr, cc, val = line_aa(startX, startY, endX, endY)
    b = data[idx]

    # Draw the line by adding the absolute pixel values (val*bgDiff) 
    # of the line segment to the background values on the data frame
    b[rr, cc] += (val * bgDiff).astype(np.int32)

    # Target (expected output) for shower frame is [1, 0]
    np.put(targets[idx], [0, 1], [1, 0])

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
