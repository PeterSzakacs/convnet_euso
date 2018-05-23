import numpy as np
import utility_funtions as utl

# import collections
# import event_visualization as vis
# from sklearn.cluster import DBSCAN
# from sklearn import metrics

cimport numpy as np
cimport cython


class NeighbourSelectionRules(object):
    max_gap = 3
    val_ratio_thr = 1.0
    grow = True

    def __init__(self, max_gap=3, val_ratio_thr=1, grow=True):
        self.max_gap = int(max_gap)
        self.val_ratio_thr = float(val_ratio_thr)
        self.grow = bool(grow)

    def __str__(self):
        return "{:d},{:.2f},{}".format(int(self.max_gap), float(self.val_ratio_thr), bool(self.grow))

    @classmethod
    def from_str(cls,s):
        p = s.split(",")
        if len(p) != 3:
            raise Exception('Unexpected string format "{}"'.format(s))  # TODO chceck why is this happening on node 15 when using ver1 data
        return NeighbourSelectionRules(int(p[0].strip()), float(p[1].strip()), utl.str2bool(p[2].strip()))


def parse_neighbour_selection_rules_str(conf_attr_str):
    for sep in (", ",";"):
        if sep in conf_attr_str:
            return [NeighbourSelectionRules.from_str(s.strip()) for s in conf_attr_str.split(sep)]
    return None

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef gray_hough_line_optimized(np.ndarray[np.float32_t, ndim=2] input_image, float line_thicknes=2, float[:] phi_linspace=np.linspace(0, np.pi, 180, dtype=np.float32)[:-1], float rho_step=1, bint fix_negative_r=False, bint flip_image=False, bint correct_phi=True):
    cdef np.ndarray[np.float32_t, ndim=2] image

    if flip_image:
        image = np.flipud(input_image)
    else:
        image = input_image


    cdef float max_distance = np.hypot(image.shape[0], image.shape[1])
    cdef unsigned long num_rho = int(np.ceil(max_distance*2/rho_step))
    cdef float rho_correction_lower = -line_thicknes + max_distance
    cdef float rho_correction_upper = line_thicknes + max_distance
    #phi_range = phi_range - np.pi / 2
    cdef np.ndarray[np.float32_t, ndim=2] acc_matrix = np.zeros((num_rho, len(phi_linspace)), dtype=np.float32)
    # rho_acc_matrix = np.zeros((num_rho, len(phi_range)))
    # nc_acc_matrix = np.zeros((num_rho, len(phi_range)))

    # phi_corr_arr = np.ones((100,len(phi_range)))

    cdef float max_acc_matrix_val = 0

    cdef float phi_corr = 1.
    cdef unsigned long phi_index
    cdef float phi
    cdef unsigned long i
    cdef unsigned long j
    cdef float rho
    cdef long long rho_index_lower
    cdef long long rho_index_upper
    cdef float phi_norm_pi_over_2

    cdef np.ndarray[np.float32_t, ndim=1] phi_corrections;

    if correct_phi:
        phi_corrections = np.zeros(len(phi_linspace), dtype=np.float32)
        for phi_index, phi in enumerate(phi_linspace):
            phi_norm_pi_over_2 = (phi - np.floor(phi/(np.pi/2))*np.pi/2)
            if phi_norm_pi_over_2 <= np.pi/4:
                phi_corr = image.shape[1] / np.sqrt(image.shape[1] ** 2 + (image.shape[1] * np.tan( phi_norm_pi_over_2 )) ** 2)
            else:
                phi_corr = image.shape[0] / np.sqrt(image.shape[0] ** 2 + (image.shape[0] * np.tan( np.pi/2 - phi_norm_pi_over_2 )) ** 2) #np.sqrt(image.shape[0] ** 2 + (image.shape[0] / np.tan( phi_norm_pi_over_2 - np.pi/4 )) ** 2) / image.shape[1]
            phi_corrections[phi_index] = phi_corr

    for i in range(0, len(image)): # row, y-axis
        for j in range(0, len(image[i])): # col, x-axis
            if image[i,j] == 0:
                continue

            for phi_index, phi in enumerate(phi_linspace):
                if correct_phi:
                    phi_corr = phi_corrections[phi_index]

                rho = j*np.cos(phi) + i*np.sin(phi)
                #
                # if rho < 0:
                #     print("rho =",rho, "phi =", phi, "phi_index =", phi_index, "i =", i, "j=", j)

                # rho_index_lower = int((rho+rho_correction_lower) // rho_step)
                # rho_index_upper = int((rho+rho_correction_upper) // rho_step)
                rho_index_lower = int(np.round((rho+rho_correction_lower) / rho_step))
                rho_index_upper = int(np.round((rho+rho_correction_upper) / rho_step))

                if rho_index_lower < 0:
                    rho_index_lower = 0

                if rho_index_upper > num_rho:
                    rho_index_upper = num_rho

                for rho_index in range(rho_index_lower,rho_index_upper):
                    acc_matrix[rho_index, phi_index] += image[i,j] * phi_corr

    return acc_matrix, max_distance, (-max_distance, max_distance, rho_step, line_thicknes), phi_linspace


cpdef gray_hough_line(np.ndarray[np.float32_t, ndim=2] input_image, float line_thicknes=2, float[:] phi_range=np.linspace(0, np.pi, 180, dtype=np.float32)[:-1], float rho_step=1, bint fix_negative_r=False, bint flip_image=False, bint correct_phi=True):
    cdef np.ndarray[np.float32_t, ndim=2] image

    if flip_image:
        image = np.flipud(input_image)
    else:
        image = input_image

    cdef float max_distance = np.hypot(image.shape[0], image.shape[1])
    cdef unsigned long num_rho = int(np.ceil(max_distance*2/rho_step))
    cdef float rho_correction_lower = -line_thicknes + max_distance
    cdef float rho_correction_upper = line_thicknes + max_distance
    #phi_range = phi_range - np.pi / 2
    cdef np.ndarray[np.float32_t, ndim=2] acc_matrix = np.zeros((num_rho, len(phi_range)), dtype=np.float32)
    # rho_acc_matrix = np.zeros((num_rho, len(phi_range)))
    # nc_acc_matrix = np.zeros((num_rho, len(phi_range)))

    # phi_corr_arr = np.ones((100,len(phi_range)))

    cdef float max_acc_matrix_val = 0

    cdef float phi_corr = 1.
    cdef unsigned long phi_index
    cdef float phi
    cdef unsigned long i
    cdef unsigned long j
    cdef float rho
    cdef long long rho_index_lower
    cdef long long rho_index_upper
    cdef float phi_norm_pi_over_2

    for phi_index, phi in enumerate(phi_range):
        # print("hough > phi = {} ({})".format(np.rad2deg(phi), phi_index))

        if correct_phi:
            phi_norm_pi_over_2 = (phi - np.floor(phi/(np.pi/2))*np.pi/2)
            if phi_norm_pi_over_2 <= np.pi/4:
                phi_corr = image.shape[1] / np.sqrt(image.shape[1] ** 2 + (image.shape[1] * np.tan( phi_norm_pi_over_2 )) ** 2)
            else:
                phi_corr = image.shape[0] / np.sqrt(image.shape[0] ** 2 + (image.shape[0] * np.tan( np.pi/2 - phi_norm_pi_over_2 )) ** 2) #np.sqrt(image.shape[0] ** 2 + (image.shape[0] / np.tan( phi_norm_pi_over_2 - np.pi/4 )) ** 2) / image.shape[1]

        # normalization vis would go here

        # phi_corr = 1 #(np.cos(phi*4) + 1)/2 + 1
        for i in range(0, len(image)): # row, y-axis
            for j in range(0, len(image[i])): # col, x-axis
                rho = j*np.cos(phi) + i*np.sin(phi)
                #
                # if rho < 0:
                #     print("rho =",rho, "phi =", phi, "phi_index =", phi_index, "i =", i, "j=", j)

                # rho_index_lower = int((rho+rho_correction_lower) // rho_step)
                # rho_index_upper = int((rho+rho_correction_upper) // rho_step)
                rho_index_lower = int(np.round((rho+rho_correction_lower) / rho_step))
                rho_index_upper = int(np.round((rho+rho_correction_upper) / rho_step))

                if rho_index_lower < 0:
                    # print("rho_index_lower < 0 : rho_index_lower=", rho_index_lower)
                    rho_index_lower = 0

                if rho_index_upper > num_rho:
                    # print("rho_index_upper > num_rho : rho_index_upper=", rho_index_upper,"num_rho=",num_rho)
                    rho_index_upper = num_rho

                for rho_index in range(rho_index_lower,rho_index_upper):
                    acc_matrix[rho_index, phi_index] += image[i,j] * phi_corr

    if not fix_negative_r:
        return acc_matrix, max_distance, (-max_distance, max_distance, rho_step, line_thicknes), phi_range
    else:
        # this is not sufficient to eliminate peak splitting due limited phi range

        if len(phi_range) < 2:
            raise Exception('Insufficient number of phi steps')

        zero_rho_index = int((0 + max_distance) // rho_step)
        zero_phi_index = len(phi_range) // 2

        acc_matrix_positive_r = acc_matrix[zero_rho_index+1:,:]
        # acc_matrix_negative_r_right = np.flipud(acc_matrix[zero_rho_index+1:,zero_phi_index:])
        # acc_matrix_negative_r_left = np.flipud(acc_matrix[zero_rho_index+1:,:zero_phi_index])
        acc_matrix_negative_r_flipped = np.flipud(acc_matrix[0:zero_rho_index])
        # acc_matrix_negative_r_right = np.flipud(acc_matrix[zero_rho_index+1:,:])
        # acc_matrix_negative_r_left = np.flipud(acc_matrix[zero_rho_index+1:,:])

        # z[0:3,1:4] = r[:3,:3]

        # print(acc_matrix.shape, acc_matrix_positive_r.shape, acc_matrix_negative_r_flipped.shape, acc_matrix_negative_r_flipped.shape)

        # vis.visualize_hough_space(acc_matrix_positive_r, phi_range, (0, max_distance, rho_step), r"Hough space with positive $\rho$ part BEFORE CORRECTION")

        max_height = max([acc_matrix_positive_r.shape[0], acc_matrix_negative_r_flipped.shape[0], acc_matrix_negative_r_flipped.shape[0]])
        if acc_matrix_positive_r.shape[0] < max_height:
            new_acc_matrix_positive_r = np.zeros((max_height, acc_matrix_positive_r.shape[1]))
            new_acc_matrix_positive_r[0:acc_matrix_positive_r.shape[0],:] = acc_matrix_positive_r
            # old = acc_matrix_positive_r
            acc_matrix_positive_r = new_acc_matrix_positive_r
            # import matplotlib.pyplot as plt
            # fig,ax = plt.subplots(1)
            # ax.imshow(old)
            # fig,ax = plt.subplots(1)
            # ax.imshow(acc_matrix_positive_r)
        elif acc_matrix_negative_r_flipped.shape[0] < max_height:
            new_acc_matrix_negative_r = np.zeros((max_height, acc_matrix_negative_r_flipped.shape[1]))
            new_acc_matrix_negative_r[0:acc_matrix_negative_r_flipped.shape[0], :] = acc_matrix_negative_r_flipped
            acc_matrix_negative_r_flipped = new_acc_matrix_negative_r

        # vis.visualize_hough_space(acc_matrix, phi_range, (-max_distance, max_distance, rho_step), r"Hough space with negative and positive $\rho$")
        #
        # vis.visualize_hough_space(np.flipud(acc_matrix_negative_r_flipped),phi_range, (-max_distance, 0, rho_step), r"Hough space with negative $\rho$ part")
        #
        # vis.visualize_hough_space(acc_matrix_positive_r, phi_range, (0, max_distance, rho_step), r"Hough space with positive $\rho$ part")

        print(acc_matrix_positive_r[:,-1:].shape, acc_matrix_positive_r[:,:1].shape)

        count_nonzero_right = np.count_nonzero(acc_matrix_positive_r[:,-1:]>0)
        count_nonzero_left = np.count_nonzero(acc_matrix_positive_r[:,:1]>0)

        l = []
        new_phi_range = phi_range
        if count_nonzero_left>=count_nonzero_right: # todo test phi_range +1
            # l.append(acc_matrix_negative_r_flipped[:,:-1])
            # new_phi_range = np.hstack(((phi_range[:-1]-phi_range[-1]),phi_range))
            l.append(acc_matrix_negative_r_flipped)
            # new_phi_range = np.hstack((new_phi_range-(2*new_phi_range[-1]-new_phi_range[-2]),new_phi_range)) # problem if not starting form 0
            # new_phi_range = np.hstack((new_phi_range-(2*new_phi_range[-1]-new_phi_range[-2])+new_phi_range[0],new_phi_range)) # problem if not starting form 0
            new_phi_range = np.hstack((new_phi_range-3*new_phi_range[-1]+2*new_phi_range[-2]+new_phi_range[0],new_phi_range)) # problem if not starting form 0
        l.append(acc_matrix_positive_r)
        if count_nonzero_left<count_nonzero_right: # TODO range needs to be corrected and solution need to be validated
            # l.append(acc_matrix_negative_r_flipped[:,1:]) # phi range should not contain pi and this might be unnecessary
            # new_phi_range = np.hstack((phi_range,(phi_range[1:]+phi_range[-1])))
            l.append(acc_matrix_negative_r_flipped)
            # new_phi_range = np.hstack((new_phi_range,new_phi_range+(2*new_phi_range[-1]-new_phi_range[-2]))) # considering pi_range[0] == 0
            new_phi_range = np.hstack((new_phi_range,new_phi_range+(new_phi_range[-1]-2*new_phi_range[-2]))) # considering pi_range[0] == 0 # this might be closer to the correct solution

        acc_matrix_fixed = np.hstack(l)

        #print(acc_matrix_fixed.shape[1] == len(new_phi_range))
        assert acc_matrix_fixed.shape[1] == len(new_phi_range)

        return acc_matrix_fixed, max_distance, (0, max_distance, rho_step, line_thicknes), new_phi_range


def normalize_phi_float(float phi):
    cdef float norm_phi = np.arctan2(np.sin(phi), np.cos(phi))
    if norm_phi < 0:
        norm_phi = (2*np.pi + norm_phi)
    # norm_phi = (2*np.pi + norm_phi) * (norm_phi < 0) + norm_phi * (norm_phi > 0)
    return norm_phi


normalize_phi_vectorized_float = np.vectorize(normalize_phi_float,otypes=(np.float32,))


def simplify_hough_space_line(float rho, float phi):
    if rho < 0:
        phi = phi - np.pi # could be negative
        rho *= -1
    # elif rho == 0:
        # phi = phi % (np.pi if phi >= 0 else -np.pi)
    # phi = phi % (2*np.pi if phi >= 0 else -2*np.pi)
    phi = normalize_phi_float(phi)
    return rho, phi


def hough_space_rho_index_to_val(index, rho_range_opts):
    return rho_range_opts[0] + rho_range_opts[2] * index + rho_range_opts[3] # // 2 # TODO justification for  `+ rho_range_opts[2]`


def hough_space_index_to_val_single(index, phi_linspace, rho_range_opts):
    return (hough_space_rho_index_to_val(index[0], rho_range_opts), phi_linspace[index[1]])


def hough_space_index_to_val(indexes, phi_linspace, rho_range_opts):
    o = []
    for index in indexes:
        o.append(hough_space_index_to_val_single(index, phi_linspace, rho_range_opts))
    return o

def hough_space_index_to_val_simp_phi(indexes, phi_linspace, rho_range_opts):
    return [simplify_hough_space_line(rho, phi) for rho, phi in hough_space_index_to_val(indexes, phi_linspace, rho_range_opts)]


def calc_line_coords(phi, rho, width, height, width_start=0, height_start=0, return_none_val=None):
    p = np.zeros((2,2),dtype=np.float32)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    if np.abs(sin_phi) > 1e-11:
        if np.abs(cos_phi) > 1e-11:
            tan_phi = sin_phi/cos_phi
            p[0,0] = rho / sin_phi # - 0/tan_phi
            p[0,1] = 0
            p[1,0] = rho / sin_phi - width / tan_phi
            p[1,1] = width

            for i in range(len(p)):
                if p[i,0] > height:
                    p[i,1] = (rho/sin_phi - height) * tan_phi
                    p[i,0] = height
                elif p[i,0] < height_start:
                    p[i,1] = (rho/sin_phi - height_start) * tan_phi
                    p[i,0] = height_start
        else:
            p[0,0] = rho
            p[0,1] = width_start
            p[1,0] = rho
            p[1,1] = width
    else:
        p[0,0] = height
        p[0,1] = rho
        p[1,0] = height_start
        p[1,1] = rho

    for i in range(len(p)):
        if p[i,0] > height or p[i,0] < height_start or p[i,1] > width or p[i,1] < width_start:
            return return_none_val

    return p


cpdef calc_line_coords_float(float phi, float rho, float width, float height, return_none_val=None):
    cdef np.ndarray[np.float32_t, ndim=2] p = np.zeros((2,2),dtype=np.float32)

    cdef float sin_phi = np.sin(phi)
    cdef float cos_phi = np.cos(phi)
    cdef tan_phi
    cdef int i

    if np.abs(sin_phi) > 1e-11:
        if np.abs(cos_phi) > 1e-11:
            tan_phi = sin_phi/cos_phi
            p[0,0] = rho / sin_phi # - 0/tan_phi
            p[0,1] = 0
            p[1,0] = rho / sin_phi - width / tan_phi
            p[1,1] = width

            for i in range(len(p)):
                if p[i,0] > height:
                    p[i,1] = (rho/sin_phi - height) * tan_phi
                    p[i,0] = height
                elif p[i,0] < 0:
                    p[i,1] = (rho/sin_phi - 0) * tan_phi
                    p[i,0] = 0
        else:
            p[0,0] = rho
            p[0,1] = 0
            p[1,0] = rho
            p[1,1] = width
    else:
        p[0,0] = height
        p[0,1] = rho
        p[1,0] = 0
        p[1,1] = rho

    for i in range(len(p)):
        if p[i,0] > height or p[i,0] < 0 or p[i,1] > width or p[i,1] < 0:
            return return_none_val

    return p


def find_pixel_clusters(image, max_gap=3):
    clusters = {}

    visited_neighbourhood = np.zeros_like(image, dtype=np.bool)

    for cluster_seed_i in range(image.shape[0]):
        for cluster_seed_j in range(image.shape[1]):
            if image[cluster_seed_i, cluster_seed_j] == 0:
                continue;

            if visited_neighbourhood[cluster_seed_i,cluster_seed_j]:
                continue

            cluster_matrix = np.zeros_like(image, dtype=np.bool)
            clusters[(cluster_seed_i,cluster_seed_j)] = cluster_matrix
            # similar to select_neighbours

            seed_points = [(cluster_seed_i, cluster_seed_j)]

            while seed_points:
                seed_i, seed_j = seed_points.pop()
                i_start = max(seed_i - max_gap, 0)
                i_end = min(seed_i + max_gap, image.shape[0])     # i_end without +1 s most probably a bug
                j_start = max(seed_j - max_gap, 0)
                j_end = min(seed_j + max_gap, image.shape[1])

                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        # if i == seed_i and j == seed_j:
                        #     continue
                        if not visited_neighbourhood[i,j]:
                            # todo add option to select only neighbours of initial seeds
                            if image[i, j] != 0:
                                seed_points.append((i,j))
                            cluster_matrix[i,j] = True
                            visited_neighbourhood[i,j] = True

    return clusters

class EmptyDataError(RuntimeWarning):
    pass

# specifying [np.float32_t, ndim=2] speeds up run ~45 times (faster than numpy-motivated version)
def find_minimal_rectangle(np.ndarray[np.float32_t, ndim=2] cluster_im):
    cdef int first_row = -1
    cdef int first_col = -1
    cdef int last_row = -1
    cdef int last_col = -1
    cdef int i,j
    # for i, j in itertools.product(range(cluster_im.shape[0]), range(cluster_im.shape[1])):
    for i in range(cluster_im.shape[0]):
        for j in range(cluster_im.shape[1]):
            if cluster_im[i, j]:
                if first_row < 0:
                    first_row = i
                last_row = i
                if first_col < 0 or j < first_col:
                    first_col = j
                if last_col < 0 or j > last_col:
                    last_col = j

    if last_row < 0:
        raise EmptyDataError()

    # if not (first_row >= 0 and last_row >=0 and first_col >= 0 and last_col >= 0):
    #     print(cluster_im.shape)
    #     print (first_row, last_row, first_col, last_col)
    # assert first_row >= 0 and last_row >=0 and first_col >= 0 and last_col >= 0
    return (first_row, last_row), (first_col, last_col)


def find_minimal_dimensions(cluster_im):
    p1, p2 = find_minimal_rectangle(cluster_im)
    first_row, last_row = p1
    first_col, last_col = p2
    return (last_row-first_row+1, last_col-first_col+1)


def find_minimal_rectangle_numpy(np.ndarray cluster_im, bint wrapover_x=False):
    cdef np.ndarray rows, cols
    cdef np.ndarray[np.int64_t] true_rows, true_cols
    cdef np.int64_t rmin, rmax, cmin, cmax, max_col_gap, gap_start_col, i
    rows = np.any(cluster_im, axis=1)
    cols = np.any(cluster_im, axis=0)
    true_rows = np.where(rows)[0]
    true_cols = np.where(cols)[0]
    if len(true_cols) == 0:
        raise EmptyDataError()
    first_row, last_row = true_rows[[0, -1]]
    first_col, last_col = true_cols[[0, -1]]

    if wrapover_x:
        max_col_gap = len(cols) - 1 - last_col + first_col
        gap_start_col = first_col+1
        for i in range(first_col+1,last_col):
            if cols[i-1] and not cols[i]:
                gap_start_col = i
            elif not cols[i-1] and cols[i]:
                gap_len = i - gap_start_col
                if gap_len > max_col_gap:
                    max_col_gap = gap_len
                    last_col = gap_start_col-1
                    first_col = -(len(cols)-i)

    return first_row, last_row, first_col, last_col


def find_minimal_dimensions_numpy(cluster_im, wrapover_x=False):
    cdef long long first_row, last_row, first_col, last_col
    first_row, last_row, first_col, last_col = find_minimal_rectangle_numpy(cluster_im, wrapover_x)
    return (last_row-first_row+1, last_col-first_col+1)


# not optimal implementation
# DEPRECATED
def select_neighbours(initial_seed_points, image, selections=[NeighbourSelectionRules(3, 1, True)]):
    # seed_points iterable of pairs
    # presuming 2d matrix

    # distance_counter reset - if examined point is seed point
    # similar of higher intensity increases search distance

    if not initial_seed_points:
        raise Exception("initial_seed_points cannot be empty")

    if len(image.shape) != 2:
        raise Exception("unexpected image shape")

    visited_neighbourhood = []
    for _ in selections:
        visited_neighbourhood.append(np.zeros_like(image, dtype=np.bool))
    out_neighbourhood = np.zeros_like(image, dtype=np.bool)

    # i - row
    # j - column

    individual_neighbourhoods = {}

    for seed_i, seed_j in initial_seed_points:
        out_neighbourhood[seed_i, seed_j] = True
        individual_neighbourhoods[(seed_i, seed_j)] = None

    seed_points = list(initial_seed_points)
    last_initial_seed_point = seed_points[-1]
    individual_neighbourhoods[last_initial_seed_point] = np.zeros_like(image)

    while seed_points:
        seed_i, seed_j = seed_points.pop()
        if (seed_i, seed_j) in individual_neighbourhoods and individual_neighbourhoods[(seed_i,seed_j)] is None:    #TODO
            last_initial_seed_point = (seed_i, seed_j)
            individual_neighbourhoods[last_initial_seed_point] = np.zeros_like(image)

        # 3 from seed included
        # v/this_v > thr => new seed

        # visited_neighbourhood[seed_i, seed_j] = True
        out_neighbourhood[seed_i, seed_j] = True
        individual_neighbourhoods[last_initial_seed_point][seed_i,seed_j] = True

        for si, selection in enumerate(selections):
            visited_neighbourhood[si][seed_i, seed_j] = True
            # out_neighbourhood[seed_i, seed_j] = True

            i_start = max(seed_i - selection.max_gap, 0)
            i_end = min(seed_i + selection.max_gap, image.shape[0])
            j_start = max(seed_j - selection.max_gap, 0)
            j_end = min(seed_j + selection.max_gap, image.shape[1])

            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    # if i == seed_i and j == seed_j:
                    #     continue
                    if not visited_neighbourhood[si][i,j]:
                        # todo add option to select only neighbours of initial seeds
                        if  (image[seed_i, seed_j]==0 or image[i,j]/image[seed_i, seed_j] > selection.val_ratio_thr) and (i,j) not in seed_points:
                            # print(image[i,j], image[seed_i, seed_j], image[i,j]/image[seed_i, seed_j])
                            if selection.grow:
                                seed_points.append((i,j))
                            out_neighbourhood[i,j] = True # prevent being added as a seed point again

    return out_neighbourhood, individual_neighbourhoods #grouped_neighbourhoods


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def select_neighbours_w_bordered_pixels(initial_seed_points, np.ndarray[np.float32_t, ndim=2] image, list selections=[NeighbourSelectionRules(3, 1, True)],
                      bint accept_zero_value_seeds=False, unsigned long pixel_value_border=0, bint include_zero_values_in_mean=False):
    # not an optimal implementation
    # seed_points iterable of pairs
    # presuming 2d matrix

    # distance_counter reset - if examined point is seed point
    # similar of higher intensity increases search distance

    # i - row
    # j - column

    # TODO 'gap' should be renamed to more suitable name - maybe border

    cdef float seed_pixel_value = -1.
    cdef float t_pixel_value = -1.

    cdef unsigned long seed_i = -1
    cdef unsigned long seed_j = -1
    cdef unsigned long i = -1
    cdef unsigned long j = -1

    if len(initial_seed_points) == 0:
        raise Exception("initial_seed_points cannot be empty")

    # if len(image.shape) != 2:
    #     raise Exception("unexpected image shape")

    cdef np.ndarray[np.uint8_t, ndim=3] visited_neighbourhood = np.zeros((len(selections), image.shape[0], image.shape[1]),dtype=np.uint8)
    # [np.bool_t, ndim=3]

    cdef np.ndarray[np.uint8_t, ndim=2] out_neighbourhood = np.zeros_like(image, dtype=np.uint8)
    # [np.bool_t, ndim=2]

    for seed_i, seed_j in initial_seed_points:
        out_neighbourhood[seed_i, seed_j] = True

    cdef list seed_points = list(initial_seed_points)

    while seed_points:
        seed_i, seed_j = seed_points.pop()

        if pixel_value_border > 0:
            pixel_value_subarray = image[
                                   min(seed_i - pixel_value_border, 0):max(seed_i + pixel_value_border + 1, image.shape[0]),
                                   min(seed_j - pixel_value_border, 0):max(seed_j + pixel_value_border + 1, image.shape[1])]
            if not include_zero_values_in_mean:
                pixel_value_subarray = pixel_value_subarray[pixel_value_subarray>0]
            seed_pixel_value = np.mean(pixel_value_subarray)
        else:
            seed_pixel_value = image[seed_i,seed_j]

        if not accept_zero_value_seeds and seed_pixel_value == 0:
            continue

        out_neighbourhood[seed_i, seed_j] = True

        for si, selection in enumerate(selections):
            visited_neighbourhood[si,seed_i,seed_j] = True
            # out_neighbourhood[seed_i, seed_j] = True

            i_start = max(seed_i - selection.max_gap, 0)
            i_end = min(seed_i + selection.max_gap + 1, image.shape[0])
            j_start = max(seed_j - selection.max_gap, 0)
            j_end = min(seed_j + selection.max_gap + 1, image.shape[1])

            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    # if i == seed_i and j == seed_j:
                    #     continue
                    if not visited_neighbourhood[si,i,j]:
                        # todo add option to select only neighbours of initial seeds

                        if pixel_value_border > 0:
                            pixel_value_subarray = image[
                                                   min(i - pixel_value_border, 0):max(i + pixel_value_border + 1, image.shape[0]),
                                                   min(j - pixel_value_border, 0):max(j + pixel_value_border + 1, image.shape[1])]
                            if not include_zero_values_in_mean:
                                pixel_value_subarray = pixel_value_subarray[pixel_value_subarray>0]
                            t_pixel_value = np.mean(pixel_value_subarray)
                        else:
                            t_pixel_value = image[i,j]

                        if (seed_pixel_value==0 or t_pixel_value/seed_pixel_value > selection.val_ratio_thr) and (i,j) not in seed_points:
                            # print(image[i,j], image[seed_i, seed_j], image[i,j]/image[seed_i, seed_j])
                            if selection.grow:
                                seed_points.append((i,j))
                            out_neighbourhood[i,j] = True # prevent being added as a seed point again

    # commnented out code is in backup.py.txt

    return out_neighbourhood


def select_trigger_groups(trigger_points, max_gap=3):
    # seed_points iterable of pairs
    # presuming 2d matrix

    # visited_neighbourhood = np.zeros_like(image, dtype=np.bool)

    trigger_groups = []

    # i - row
    # j - column

    point_neighbours = {}
    visited_points = {}
    for trigger_point in trigger_points:
        point_neighbours[trigger_point] = []
        visited_points[trigger_point] = False

    for trigger_point in trigger_points:
        for c_trigger_point in trigger_points: # very inefficient
            if c_trigger_point != trigger_point and \
                    abs(trigger_point[0] - c_trigger_point[0]) <= max_gap and \
                    abs(trigger_point[1] - c_trigger_point[1]) <= max_gap:
                point_neighbours[trigger_point].append(c_trigger_point)
            # if group is None:
            #     group = [trigger_point]
            #     trigger_groups.append(group)

    for trigger_point in trigger_points:
        if visited_points[trigger_point]:
            continue

        visited_points[trigger_point] = True

        group = [trigger_point]
        trigger_groups.append(group)

        search_stack = list(point_neighbours[trigger_point])
        while search_stack:
            neighbour_point = search_stack.pop()
            if not visited_points[neighbour_point]:
                visited_points[neighbour_point] = True
                group.append(neighbour_point)
                for neighbour_neighbour_point in point_neighbours[neighbour_point]:
                    if neighbour_neighbour_point != trigger_point and not visited_points[neighbour_neighbour_point]:
                        search_stack.append(neighbour_neighbour_point)

    return trigger_groups

    # for i in range(0,len(image.shape[0]): # row

    # it = np.nditer(a, flags=['multi_index'])
    # while not it.finished:
    #     ...
    #     print
    #     "%d <%s>" % (it[0], it.multi_index),
    #     ...
    #     it.iternext()


def normalize_phi_float(float phi):
    cdef float norm_phi = np.arctan2(np.sin(phi), np.cos(phi))
    norm_phi = (2*np.pi + norm_phi) * (norm_phi < 0) + norm_phi * (norm_phi > 0)
    return norm_phi


def normalize_hough_space_line_float(float rho, float phi):
    if rho < 0:
        phi = phi - np.pi # could be negative
        rho *= -1
    # elif rho == 0:
        # phi = phi % (np.pi if phi >= 0 else -np.pi)
    # phi = phi % (2*np.pi if phi >= 0 else -2*np.pi)
    phi = normalize_phi_float(phi)
    return rho, phi


def hs_rho_index2val(index, rho_range_opts):
    return rho_range_opts[0] + rho_range_opts[2] * index + rho_range_opts[2]*0.5  # or? # + rho_range_opts[3]/2
    # (0, max_distance, rho_step, line_thicknes)


def hs_index2val(index, phi_linspace, rho_range_opts):
    return (hs_rho_index2val(index[0], rho_range_opts), phi_linspace[index[1]])


def hs_index2val_normalized_single_float(index, phi_linspace, rho_range_opts):
    return normalize_hough_space_line_float(hs_rho_index2val(index[0], rho_range_opts), phi_linspace[index[1]])


def hs_indexes2val(indexes, phi_linspace, rho_range_opts):
    # this should return/use nd.array
    cdef list o = []
    # cdef tuple index
    for index in indexes:
        o.append(hs_index2val(index, phi_linspace, rho_range_opts))
    return o


# make_positive_angles = np.vectorize(lambda ang: ang+2*np.pi if ang < 0 else ang)


def hs_indexes2val_normalized_float(indexes, phi_linspace, rho_range_opts):
    return [normalize_hough_space_line_float(rho, phi) for rho, phi in hs_indexes2val(indexes, phi_linspace, rho_range_opts)]


def wrap_over_index(long n, unsigned int llen):
    return n-llen if n >= llen else (n if n >= 0 else llen+n)


def angle_difference_float(float a1, float a2):
    cdef float r = np.abs(a1 - a2) % (2*np.pi)
    if r >= np.pi:
        r = 2*np.pi - r
    return r


def circular_range(num_elements, low_index, up_index=None, step=-1):
    # i != up_index
    if up_index is None:
        up_index = low_index

    it_1_up = -1
    it_2_low = num_elements -1

    if step > 0:
        if low_index < up_index:
            it_1_up = up_index
            it_2_low = None
        else:
            it_1_up = num_elements
            it_2_low = 0
    else:
        if low_index < up_index:
            pass # it_1_up = -1
            # it_2_low = num_elements -1
        else:
            it_1_up = up_index
            it_2_low = None

    for i in range(low_index, it_1_up, step):
        yield i

    if it_2_low is not None:
        for i in range(it_2_low, up_index, step):
            yield i


def invert_line(float rho, float phi):
    phi -= np.pi
    rho *= -1
    return rho, phi


def hough_space_distance_value_float(float a_rho, float a_phi, float b_rho, float b_phi):
    cdef float phi_diff
    # normalize to 0, pi range
    if a_phi < 0 or a_phi > np.pi:
        a_rho, a_phi = invert_line(a_rho, normalize_phi_float(a_phi))
    if b_phi < 0 or b_phi > np.pi:
        b_rho, b_phi = invert_line(b_rho, normalize_phi_float(b_phi))
    # angle difference considering pi wraparound
    phi_diff = np.abs(a_phi - b_phi)
    if phi_diff > np.pi/2:
        if a_phi > np.pi/2:
            a_rho, a_phi = invert_line(a_rho, a_phi)
        if b_phi > np.pi/2:
            a_rho, a_phi = invert_line(a_rho, a_phi)
    return np.hypot(a_rho - b_rho, a_phi - b_phi, dtype=np.float32)


def hough_space_distance_float(float[:] phi_linspace, tuple rho_range_opts,
                               unsigned long a_i, unsigned long a_j, unsigned long b_i, unsigned long b_j):
    cdef float a_rho
    cdef float a_phi
    cdef float b_rho
    cdef float b_phi
    a_rho, a_phi = hs_index2val((np.int64(a_i), np.int64(a_j)), phi_linspace, rho_range_opts)
    # int64 is to manage float64 conversion that happens in scipy
    b_rho, b_phi = hs_index2val((np.int64(b_i), np.int64(b_j)), phi_linspace, rho_range_opts)
    return hough_space_distance_value_float(a_rho, a_phi, b_rho, b_phi)


# distance_func causes inefficiency
# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_pixel_clusters_in_image_float(np.ndarray[np.float32_t, ndim=2] image, max_box_gap=3, distance_func=None,
                                       bint wrapover_0=False, bint wrapover_1=True,
                                       bint inverse_1_on_wrapover_of_0=False, bint inverse_0_on_wrapover_of_1=True):
    cdef int max_gap_i, max_gap_j, cluster_seed_i, cluster_seed_j, i_wrapover_j_index, j_wrapover_i_index, i, ii, j, jj, oi, oj
    cdef bint i_start_wrapovered, j_start_wrapovered

    # cdef np.ndarray cluster_matrix
    cdef np.ndarray[np.uint8_t, ndim=2] cluster_matrix
    # considers pi width of image
    cdef dict clusters = {}
    cdef list seed_points
    # cdef np.ndarray visited_neighbourhood = np.zeros_like(image, dtype=np.bool)
    cdef np.ndarray[np.uint8_t, ndim=2] visited_neighbourhood = np.zeros_like(image, dtype=np.uint8)

    if not wrapover_0:
        inverse_1_on_wrapover_of_0 = False

    if not wrapover_1:
        inverse_0_on_wrapover_of_1 = False

    if isinstance(max_box_gap, (list, tuple)):
        max_gap_i = max_box_gap[0]
        max_gap_j = max_box_gap[1]
    else:
        max_gap_i = max_gap_j = max_box_gap

    for cluster_seed_i in range(image.shape[0]):
        for cluster_seed_j in range(image.shape[1]):
            if image[cluster_seed_i, cluster_seed_j] == 0:
                continue

            if visited_neighbourhood[cluster_seed_i,cluster_seed_j]:
                continue

            cluster_matrix = np.zeros_like(image, dtype=np.uint8)
            clusters[(cluster_seed_i,cluster_seed_j)] = cluster_matrix

            cluster_matrix[cluster_seed_i, cluster_seed_j] = True
            visited_neighbourhood[cluster_seed_i, cluster_seed_j] = True

            seed_points = [(cluster_seed_i, cluster_seed_j)]

            while seed_points:
                seed_i, seed_j = seed_points.pop()

                if wrapover_0:
                    i_range = circular_range(image.shape[0],
                                             wrap_over_index(seed_i - max_gap_i, image.shape[0]),
                                             wrap_over_index(seed_i + max_gap_i + 1, image.shape[0]), 1)
                else:
                    i_range = range(max(seed_i - max_gap_i, 0), min(seed_i + max_gap_i + 1, image.shape[0]))

                if wrapover_1:
                    j_range = list(circular_range(image.shape[1],
                                             wrap_over_index(seed_j - max_gap_j, image.shape[1]),
                                             wrap_over_index(seed_j + max_gap_j + 1, image.shape[1]), 1))
                else:
                    j_range = list(range(max(seed_j - max_gap_j, 0), min(seed_j + max_gap_j + 1, image.shape[1])))

                i_wrapover_j_index = 0
                if inverse_0_on_wrapover_of_1:
                    i_start_wrapovered = seed_j < max_gap_j
                    if i_start_wrapovered:
                        i_wrapover_j_index = max_gap_j - seed_j
                    i_end_wrapovered = seed_j+max_gap_j > image.shape[1]
                    if i_end_wrapovered:
                        i_wrapover_j_index = image.shape[1] + max_gap_j - seed_j
                else:
                    i_start_wrapovered = i_end_wrapovered = False

                j_wrapover_i_index = 0
                if inverse_1_on_wrapover_of_0:
                    j_start_wrapovered = seed_i < max_gap_i
                    if j_start_wrapovered:
                        j_wrapover_i_index = max_gap_i - seed_i
                    j_end_wrapovered = seed_i+max_gap_i > image.shape[0]
                    if j_end_wrapovered:
                        j_wrapover_i_index = image.shape[1] + max_gap_i - seed_i
                else:
                    j_start_wrapovered = j_end_wrapovered = False

                for ii, i in enumerate(i_range):
                    inverse_j = (j_start_wrapovered and ii < j_wrapover_i_index) \
                                or (j_end_wrapovered and ii > j_wrapover_i_index)

                    for jj, j in enumerate(j_range):
                        if (i_start_wrapovered and jj < i_wrapover_j_index) \
                                or (i_end_wrapovered and jj > i_wrapover_j_index):
                            oi = image.shape[0] - 1 - i
                        else:
                            oi = i

                        if inverse_j:
                            oj = image.shape[1] - 1 - j
                        else:
                            oj = j

                        if not visited_neighbourhood[oi,oj]:
                            if image[oi, oj] != 0 and (distance_func(seed_i, seed_j, oi, oj) if distance_func else True):
                                seed_points.append((oi,oj))
                                cluster_matrix[oi,oj] = True
                                visited_neighbourhood[oi,oj] = True

    return clusters

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def find_major_orientation(np.ndarray[np.float32_t] phi_angles, np.ndarray[np.float32_t] line_weights):
    cdef int i_p1, i_p2, num_angles, t_i_p2, i_last_neighboring_line, t_i_last_neighboring_line
    cdef float p1, p2, last_p2, t_phi_group_sum, max_phi_group_sum
    cdef bint single_p1, d_cond
    cdef np.ndarray[np.float32_t] norm_phi, t_phi_group, sorted_phi_angles
    cdef np.ndarray[np.int64_t] line_indexes

    if len(phi_angles) == 1:
        return [0]
    if len(phi_angles) == 0:
        raise RuntimeError('phi_angles is empty')

    norm_phi = normalize_phi_vectorized_float(phi_angles)

    # sorted_phi_angles, line_indexes = zip(*sorted(zip(norm_phi, range(0,len(norm_phi))), key=lambda pair: pair[0]))
    line_indexes = np.argsort(norm_phi)
    sorted_phi_angles = norm_phi[line_indexes]
    num_angles = len(sorted_phi_angles)

    max_phi_group_sum = 0
    max_phi_group = None

    # groups = []  # just for debugging

    # t_phi_group = np.zeros(num_angles, dtype=np.bool)  # might be optimized to a bit-array
    # t2_phi_group = np.zeros(num_angles, dtype=np.bool)  # might be optimized to a bit-array
    # t_phi_group_w
    # t2_phi_group_w

    for i_p1 in range(len(sorted_phi_angles)):
        p1 = sorted_phi_angles[i_p1]

        t_phi_group = np.zeros(num_angles, dtype=np.float32)
        # t_phi_group[:] = 0
        t_phi_group[i_p1] = line_weights[i_p1]
        # t_phi_groups = [t_phi_group]

        ###########
        # groups.append(t_phi_group)
        # if i_p1==100:
        #     print(i_p1)
        # if len(groups)>=671:
        #     print('len(groups)==2244')
        ###########

        # last_p2 = p1
        # pass over neighboring points following p1, max distance < 90
        for i_p2 in range(i_p1+1, len(sorted_phi_angles)):
            p2 = sorted_phi_angles[i_p2]
            # if angle_difference_float(p1, p2) > np.pi/2: #or angle_difference_float(last_p2,p2) > np.pi/2:
            if p2 - p1 > np.pi/2:
                i_p2 -= 1 # i_p2 should be last appended
                break
            t_phi_group[i_p2] = line_weights[i_p2]
            # last_p2 = p2

        # check points following p1 wrapped-around the unit circle
        if 2*np.pi - sorted_phi_angles[i_p1] < np.pi/2:
            last_p2 = p1
            for t_i_p2 in range(0, i_p1):
                p2 = sorted_phi_angles[t_i_p2]
                # if angle_difference_float(p1, p2) > np.pi/2 or angle_difference_float(last_p2,p2) > np.pi/2:
                if p2 > np.pi/2 or angle_difference_float(p1, p2) > np.pi/2:
                    i_p2 = t_i_p2 -1
                    break
                i_p2 = t_i_p2
                t_phi_group[i_p2] = line_weights[i_p2]
                last_p2 = p2

        if angle_difference_float(p1, sorted_phi_angles[i_p2]) <= np.pi / 2:
            i_last_neighboring_line = i_p2

        single_p1 = i_last_neighboring_line == i_p1

        # if i_last_neighboring_line == i_p1:
        #     i_last_neighboring_line = wrap_over_index(i_p1 - 1, num_angles)

        # pass over opposite points, max distance > 90 and  > 90 from nearest neighbor
        if np.count_nonzero(t_phi_group) != num_angles:
            for i_p2 in circular_range(num_angles, wrap_over_index(i_p1 - 1, num_angles), i_p1, -1):
                # if i_p2 >= 43:
                #     print("i_p2=",i_p2)

                p2 = sorted_phi_angles[i_p2]
                if angle_difference_float(p1, p2) <= np.pi/2:
                    continue
                else:
                    d_cond = single_p1 or \
                             angle_difference_float(sorted_phi_angles[i_last_neighboring_line], p2) > np.pi / 2
                    if not d_cond:
                        t = t_i_last_neighboring_line = i_last_neighboring_line
                        while t_i_last_neighboring_line != i_p1:
                            t_i_last_neighboring_line = wrap_over_index(t_i_last_neighboring_line - 1, num_angles)
                            if t_i_last_neighboring_line == i_p1:
                                break
                            if angle_difference_float(sorted_phi_angles[t_i_last_neighboring_line], p2) > np.pi / 2:
                                d_cond = True
                                break
                            t = t_i_last_neighboring_line

                        if d_cond:
                            t_phi_group_sum = np.sum(t_phi_group)
                            if t_phi_group_sum > max_phi_group_sum:
                                max_phi_group = t_phi_group
                                max_phi_group_sum = t_phi_group_sum

                            t_phi_group = np.copy(t_phi_group)
                            for i_n in circular_range(num_angles, t, i_last_neighboring_line + 1, 1):
                                t_phi_group[i_n] = 0
                            # t_phi_groups.append(t_phi_group)

                            ##############
                            # groups.append(t_phi_group)
                            # if len(groups) >= 671:
                            #     print('len(groups)==2244')
                            ##############

                        i_last_neighboring_line = t_i_last_neighboring_line

                    if d_cond:
                        t_phi_group[i_p2] = line_weights[i_p2]

                if not single_p1 and i_last_neighboring_line == i_p1:
                    break

            if np.count_nonzero(t_phi_group) == num_angles:
                # groups.append(tuple(t_phi_group))
                max_phi_group = t_phi_group
                break
        else:
            # groups.append(tuple(t_phi_group))
            max_phi_group = t_phi_group
            break

        t_phi_group_sum = np.sum(t_phi_group)
        if t_phi_group_sum > max_phi_group_sum:
            max_phi_group = t_phi_group
            max_phi_group_sum = t_phi_group_sum

    return sorted([line_indexes[i_p] for i_p, w in enumerate(max_phi_group) if w > 0])
