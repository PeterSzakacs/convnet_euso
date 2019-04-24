import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np


# metadata to text conversion


def _get_synth_bg_text(metadata):
    bec = metadata['num_bad_ECs']
    bg_lam = float(metadata['bg_lambda'])
    bg_lam = round(bg_lam, 2)
    return "bg avg: {}, {} bad EC modules".format(bg_lam, bec)


def _get_synth_shower_text(metadata):
    angle = metadata.get('yx_angle', None)
    maximum = metadata.get('shower_max', None)
    duration = metadata.get('duration', None)
    return "inclination {}°, lasting {} GTUs, peak {} PE counts".format(
        angle, duration, maximum
    )


def flight_metadata_to_text(metadata):
    eid = metadata.get('event_id')
    if eid is None:
        src = metadata['source_file_acquisition_full']
        idx = metadata['packet_id']
        return "From packet {} in source file \n{}".format(idx, src)
    else:
        return 'event id: {}'.format(eid)


def simu_metadata_to_text(metadata):
    eid = metadata.get('event_id')
    if eid is None:
        src = metadata['source_file_acquisition_full']
        idx = metadata['packet_id']
        return "From packet {} in source file \n{}".format(idx, src)
    else:
        return 'event id: {}'.format(eid)


def synth_metadata_to_text(metadata):
    bg_text = _get_synth_bg_text(metadata)
    if metadata['shower'] == 'True':
        shower_text = _get_synth_shower_text(metadata)
        return ("Shower {} ({})".format(shower_text, bg_text))
    else:
        return "Pure noise ({})".format(bg_text)


# figure creation


def _create_projection_figure(frame):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(frame)
    fig.colorbar(im)
    return ax


def create_yx_proj(frame):
    ax = _create_projection_figure(frame)
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    return ax


def create_gtux_proj(frame):
    ax = _create_projection_figure(np.transpose(frame))
    ax.set_ylabel('x [pix]')
    ax.set_xlabel('time [GTU]')
    return ax


def create_gtuy_proj(frame):
    ax = _create_projection_figure(np.transpose(frame))
    ax.set_ylabel('y [pix]')
    ax.set_xlabel('time [GTU]')
    return ax


def add_flight_metadata(ax, proj_type, metadata):
    ax.set_title(flight_metadata_to_text(metadata))


def add_simu_metadata(ax, proj_type, metadata):
    ax.set_title(simu_metadata_to_text(metadata))


def add_synth_metadata(ax, proj_type, metadata):
    # any attribute other than bg_labmda will do
    if metadata['start_gtu'] != '':
        ax.set_title(_get_synth_shower_text(metadata) + "\n" +
                     "({})".format(_get_synth_bg_text(metadata)))
        # mark start position of shower with a red dot
        start_x = metadata.get('start_x', None)
        start_y = metadata.get('start_y', None)
        start_gtu = metadata.get('start_gtu', None)
        if proj_type == 'yx':
            plt.scatter([int(start_x)], [int(start_y)], color='red', s=40)
        elif proj_type == 'gtux':
            plt.scatter([int(start_gtu)], [int(start_x)], color='red', s=40)
        elif proj_type == 'gtuy':
            plt.scatter([int(start_gtu)], [int(start_y)], color='red', s=40)
        else:
            raise Exception("Illegal projection type: {}".format(proj_type))
    else:
        ax.set_title(_get_synth_bg_text(metadata))

def save_item(ax, save_pathname):
    figure = ax.figure
    figure.savefig(save_pathname)
    plt.close(figure)
