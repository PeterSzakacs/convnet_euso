import os

import utils.dataset_utils as ds
import visualization.event_visualization as eviz


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_visualizer as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    savedir = os.path.join(args.outdir, 'img')
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if args.flight:
        meta_adder = eviz.add_flight_metadata
    elif args.simu:
        meta_adder = eviz.add_simu_metadata
    elif args.synth:
        meta_adder = eviz.add_synth_metadata

    frame_creators = {
        'raw': lambda frame: None, 'yx': eviz.create_yx_proj,
        'gtux': eviz.create_gtux_proj, 'gtuy': eviz.create_gtuy_proj
    }

    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name,
                                            item_types=args.item_types)
    data = dataset.get_data_as_dict(slice(args.num_items))
    targets = dataset.get_targets(slice(args.num_items))
    metadata = dataset.get_metadata(slice(args.num_items))
    for item_type, data_items in data.items():
        item_dir = os.path.join(savedir, item_type)
        os.mkdir(item_dir)
        frame_creator = frame_creators[item_type]
        for idx in range(len(data_items)):
            frame = data_items[idx]
            fig, ax = frame_creator(frame)
            meta_adder(fig, ax, item_type, metadata[idx])
            eviz.save_figure(fig, os.path.join(item_dir, 'frame-{}'.format(idx)))