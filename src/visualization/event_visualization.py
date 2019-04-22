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
    maximum = metadata.get('max', None)
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
    plt.colorbar(im)
    return fig, ax


def create_yx_proj(frame):
    fig, ax = _create_projection_figure(frame)
    ax.set_xlabel('x [pix]')
    ax.set_ylabel('y [pix]')
    return fig, ax


def create_gtux_proj(frame):
    fig, ax = _create_projection_figure(np.transpose(frame))
    ax.set_ylabel('x [pix]')
    ax.set_xlabel('time [GTU]')
    return fig, ax


def create_gtuy_proj(frame):
    fig, ax = _create_projection_figure(np.transpose(frame))
    ax.set_ylabel('y [pix]')
    ax.set_xlabel('time [GTU]')
    return fig, ax


def add_flight_metadata(fig, ax, proj_type, metadata):
    ax.set_title(flight_metadata_to_text(metadata))


def add_simu_metadata(fig, ax, proj_type, metadata):
    ax.set_title(simu_metadata_to_text(metadata))


def add_synth_metadata(fig, ax, proj_type, metadata):
    if metadata['shower'] == 'True':
        fig.suptitle("Shower")
        ax.set_title(_get_synth_shower_text(metadata) + "\n" +
                     "({})".format(_get_synth_bg_text(metadata)))
        start_x = metadata.get('start_x', None)
        start_y = metadata.get('start_y', None)
        start_gtu = metadata.get('start_gtu', None)
        if proj_type == 'yx':
            plt.scatter([int(start_x)], [int(start_y)], color='red', s=40)
        elif proj_type == 'gtux':
            plt.scatter([int(start_x)], [int(start_gtu)], color='red', s=40)
        elif proj_type == 'gtuy':
            plt.scatter([int(start_y)], [int(start_gtu)], color='red', s=40)
        else:
            raise Exception("Illegal projection type: {}".format(proj_type))
    else:
        fig.suptitle("Pure noise")
        ax.set_title(_get_synth_bg_text(metadata))

def save_figure(figure, save_pathname):
    figure.savefig(save_pathname)
    plt.close(figure)

# if __name__ == '__main__':
#     import numpy as np

#     print(matplotlib.matplotlib_fname())

#     fig, ax = create_projection_figure(np.random.poisson(lam=1, size=(48, 48)), 'yx')
#     add_synth_metadata(fig, ax, 'yx', {'shower': True, 'start_x': 10, 'start_y': 3, 'start_gtu': 6})

#     plt.savefig("test")
#     plt.close()
