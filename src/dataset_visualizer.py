import os

import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

import utils.dataset_utils as ds

def visualize_frame(frame, frame_type, metadata, idx, outdir):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(frame)
    plt.colorbar(im)

    if metadata['shower']:
        start_x = metadata.get('start_x', None)
        start_y = metadata.get('start_y', None)
        start_gtu = metadata.get('start_gtu', None)
        if key == 'yx':
            plt.scatter([int(start_x)], [int(start_y)], color='red', s=40)
        elif key == 'gtux':
            plt.scatter([int(start_x)], [int(start_gtu)], color='red', s=40)
        else:
            plt.scatter([int(start_y)], [int(start_gtu)], color='red', s=40)

        angle = metadata.get('yx_angle', None)
        maximum = metadata.get('max', None)
        duration = metadata.get('duration', None)
        title = 'Shower (angle: {}, maximum: {}, duration: {})'.format(
            angle, maximum, duration)
    else:
        title = 'Noise'

    ax.set_title(title)
    filename = '{}_{}'.format(frame_type, idx)
    plt.savefig(os.path.join(outdir, filename))
    plt.close()

if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_visualizer as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name, 
                                            item_types=args.item_types)
    data = dataset.get_data_as_dict(slice(args.num_items))
    targets = dataset.get_targets(slice(args.num_items))
    metadata = dataset.get_metadata(slice(args.num_items))
    for idx in range(len(metadata)):
        for key, data_items in data.items():
            frame = data_items[idx]
            visualize_frame(frame, key, metadata[idx], idx, args.outdir)