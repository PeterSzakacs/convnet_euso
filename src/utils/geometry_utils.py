import math


def trim_to_packet_template(line, packet_template):
    """
        Trim the provided line to the dimensions of the packet template.

        Parameters
        ----------
        :param line:            The points on a line reprresented as lists
                                of GTU, Y and X coordinates, respectively.
        :type line:             ((int, ), (int, ), (int, ))
        :param packet_template: The points on a line reprresented as lists
                                of GTU, Y and X coordinates, respectively
        :type packet_template:  utils.data_templates.PacketTemplate
    """
    GTUs, Ys, Xs = line[:]
    num_frames = packet_template.num_frames
    height = packet_template.frame_height
    width = packet_template.frame_width

    # remove those coordinates that are outside the edges of a packet
    new_line = ([], [], [])
    line_length = len(GTUs)
    for idx in range(line_length):
        gtu, y, x = GTUs[idx], Ys[idx], Xs[idx]
        if (gtu >= 0 and gtu < num_frames and x >= 0 and x < width 
            and y >= 0 and y < height):
            new_line[0].append(gtu)
            new_line[1].append(y)
            new_line[2].append(x)
    return new_line


def get_line_end(start, yx_angle, length, duration):
    """
        Calculate the endpoint of a line in a 3D matrix given its start
        position yx angle and length in the x y and z axes.

        Parameters
        ----------
        :param start:       The start position of the line in the 3D matrix as
                            a tuple of (Z, Y, X) coordinates.
        :type start:        (int, int, int)
        :param yx_angle:    The angle in degrees of the line as viewed on a yx
                            projection of the 3D matrix.
        :type yx_angle:     float
        :param length:      The length of the line as viewed on a yx projection
                            of the 3D matrix.
        :type length:       int
        :param duration:    The length of the line as viewed in the zx and zy
                            projections of the 3D matrix (number of consecutive
                            2D matrices that the line appears in).
        :type duration:     int
    """
    ang_rad = math.radians(yx_angle)
    delta_x, delta_y = math.cos(ang_rad), math.sin(ang_rad)
    z1, y1, x1 = start[:]
    return (z1 + duration, y1 + round(delta_y*length),
            round(x1 + delta_x*length))


# reference implementation just with mildly changed output format,
# courtesy of https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
# do not test
def draw_line_bressenham(start, end):
    """
        Create a line from start to end in a 3D coordinate space using
        Bresenham's line algorithm.

        Parameters
        ----------
        :param start:   The start position of the line in as a tuple of
                        (Z, Y, X) coordinates.
        :type start:    (int, int, int)
        :param end:     The end position of the line as a tuple of
                        (Z, Y, X) coordinates.
        :type end:      (int, int, int)
    """
    z1, y1, x1 = start[:]
    z2, y2, x2 = end[:]
    ListOfPoints = [[z1],[y1],[x1]]
    def append(z, y, x):
        ListOfPoints[0].append(z1)
        ListOfPoints[1].append(y1)
        ListOfPoints[2].append(x1)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            append(z1, y1, x1)

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            append(z1, y1, x1)

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            append(z1, y1, x1)
    return ListOfPoints 