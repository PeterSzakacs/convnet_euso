import itertools
import math
import random as rand

import utils.geometry_utils as gutils

def create_simu_shower_line(yx_angle, start_coordinate, packet_template,
                            values_generator):
    """
        Create a simulated shower line and return its coordinates (GTU, X, Y)
        and values in the form of tuples of ints

        The shower line is inclined under yx_angle and starts at a given GTU,
        X and Y coordinate. Vals_generator is used to generate the values for
        every frame or GTU of am packet.

        The values and coordinates will be trimmed if necessary to conform
        to the dimensions of a packet specified through the packet_template
        parameter (to prevent going outside the number of packet frames or
        frame width or height).

        The returned tuples of coordinates and values can be used directly to
        draw the line into the actual packet.

        Parameters
        ----------
        start_coordinate :  tuple of int
                            start coordinates from which to start the shower
                            line in the form (start_gtu, start_y, start_x).
        yx_angle :          float.
                            angle in degrees under which the shower track
                            appears in the xy projection of the packet.
        packet_template :   utils.data_templates.packet_template.
                            template of the packet in which this simulated
                            shower occurs. This is to shave off any pixels
                            that are outside the packet boundaries.
        vals_generator :    iterable returning int
                            iterable object returning the desired shower value
                            at each ieration. MUST BE PRESET TO GENERATE THE
                            CORRECT VALUES FOR EVERY GTU.

        Returns
        -------
        GTU :               tuple of int
                            tuple, where a given element represents a GTU-axis
                            coordinate for a particular shower value at the
                            same index in Values.
        Y :                 tuple of int
                            tuple, where a given element represents a Y-axis
                            coordinate for a particular shower value at the
                            same index in Values.
        X :                 tuple of int
                            tuple, where a given element represents an X-axis
                            coordinate for a particular shower value at the
                            same index in Values.
        Values:             tuple of int
                            tuple of shower values in ordered with respect to
                            time (GTU)
    """
    ang_rad = math.radians(yx_angle)
    delta_x, delta_y = math.cos(ang_rad), math.sin(ang_rad)

    start_gtu, start_y, start_x = start_coordinate[0:3]
    num_frames = packet_template.num_frames
    height = packet_template.frame_height
    width = packet_template.frame_width

    # generate shower values for every GTU from start_gtu to num_frames or
    # until the shower generator finishes
    Values = [val for val in itertools.islice(values_generator,
                                              num_frames - start_gtu)]
    GTU = range(start_gtu, start_gtu + len(Values))
    # get x, y coordinates for every gtu
    X = [start_x + math.floor(delta_x * idx) for idx in range(len(Values))]
    Y = [start_y + math.floor(delta_y * idx) for idx in range(len(Values))]
    # remove those that are outside the edge of a packet frame
    X = [x for x in X if x >= 0 and x < width]
    Y = [y for y in Y if y >= 0 and y < height]
    num_frames = min(len(X), len(Y))
    X, Y = tuple(X[:num_frames]), tuple(Y[:num_frames])
    GTU, Values = tuple(GTU[:num_frames]), tuple(Values[:num_frames])
    return GTU, Y, X, Values


def create_simu_shower_line_from_template(shower_template, yx_angle,
                                              return_metadata=False):
    start = shower_template.get_new_start_coordinate()
    shower_max = shower_template.get_new_shower_max()
    duration = shower_template.get_new_shower_duration()
    length = shower_template.get_new_track_length()

    packet_template = shower_template.packet_template
    end = gutils.get_line_end(start, yx_angle, length, duration)
    line = gutils.draw_line_bressenham(start, end)
    line = gutils.trim_to_packet_template(line, packet_template)

    GTU, Y, X = line[:]
    vals_generator = shower_template.values_generator
    vals_generator.reset(shower_max, len(GTU))
    Vals = tuple(val for val in vals_generator)

    if return_metadata:
        angle = round(yx_angle % 360)
        return GTU, Y, X, Vals, {"start_gtu": start[0], "start_y": start[1],
                                 "start_x": start[2], "duration": len(Vals),
                                 "max": max(Vals), "yx_angle": angle,
                                 "length": length}
    else:
        return GTU, Y, X, Vals


def create_simu_shower_line_from_template_old(shower_template, yx_angle,
                                          return_metadata=False):
    start = shower_template.get_new_start_coordinate()
    shower_max = shower_template.get_new_shower_max()
    duration = shower_template.get_new_shower_duration()
    vals_generator = shower_template.values_generator
    vals_generator.reset(shower_max, duration)

    packet_template = shower_template.packet_template
    GTU, Y, X, Vals = create_simu_shower_line(yx_angle, start, packet_template,
                                              vals_generator)
    if return_metadata:
        angle = round(yx_angle % 360)
        return GTU, Y, X, Vals, {"start_gtu": start[0], "start_y": start[1],
                                 "start_x": start[2], "duration": len(Vals),
                                 "max": max(Vals), "yx_angle": angle}
    else:
        return GTU, Y, X, Vals


def select_random_ECs(packet_template, max_ECs, excluded_ECs=[]):
    """
        Randomly select regions of a packet frame corresponding to the EC units
        on the surface of a source detector (can be used to e.g. simulate
        malfunctioning EC units of a detector).

        The number of regions or ECs actually selected is the minimum of
        (max_ECs) or (all possible ECs minus those in excluded_ECs).

        Parameters
        ----------
        packet_template :   utils.packets.packet_utils.packet_template
                            template of packet data
        max_errors :        int
                            maximum number of ECs to select.
        excluded_ECs :      (list or tuple) of ints
                            EC indexes that should not be selected.

        Returns
        -------
        X :                 tuple of slices
                            tuple, where a given element represents the X-axis
                            slice for an EC whose EC index is in EC_indices at
                            the same index as the slice.
        Y :                 tuple of slices
                            tuple, where a given element represents the Y-axis
                            slice for an EC whose EC index is in EC_indices at
                            the same index as the slice.
        EC_indices :        tuple of int
                            EC indexes of selected ECs.
    """
    EC_n = packet_template.num_EC
    indices = set(range(0, EC_n, 1)).difference(excluded_ECs)
    used_indices = set()
    X, Y = [], []
    stop = min(len(indices), max_ECs)
    for iteration in range(0, stop):
        index = rand.randrange(0, EC_n)
        # do not use an EC index that was explicitly excluded nor one that was
        # already used
        while index not in indices or index in used_indices:
            index = rand.randrange(0, EC_n)
        x, y = packet_template.ec_idx_to_xy_slice(index)
        # perhaps better to not store slices but the actual X and Y positions
        X.append(x), Y.append(y)
        used_indices.add(index)
        indices.discard(index)
    return tuple(X), tuple(Y), tuple(used_indices)
