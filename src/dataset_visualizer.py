import os

import dataset.dataset_utils as ds
import utils.io_utils as io_utils
import visualization.event_visualization as eviz

frame_creators = {
    'raw': lambda frame: None,
    'yx': eviz.create_yx_proj,
    'gtux': eviz.create_gtux_proj,
    'gtuy': eviz.create_gtuy_proj
}

meta_to_text = {
    'simu': eviz.add_simu_metadata,
    'flight': eviz.add_flight_metadata,
    'synth': eviz.add_synth_metadata
}

if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_visualizer as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args = ui.get_cmd_args(sys.argv[1:])
    print(args)

    name, srcdir = args.name, args.srcdir
    item_types = args.item_types
    outdir = args.outdir

    savedir = os.path.join(outdir, 'img')
    os.makedirs(savedir, exist_ok=args.force_overwrite)

    meta_adder = meta_to_text[args.meta_to_text_conv]

    handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    dataset = handler.load_dataset(name, item_types=item_types)

    start, stop = args.start_item, args.stop_item
    items_slice = slice(start, stop)
    data = dataset.get_data_as_dict(items_slice)
    targets = dataset.get_targets(items_slice)
    metadata = dataset.get_metadata(items_slice)
    for item_type, data_items in data.items():
        item_dir = os.path.join(savedir, item_type)
        os.makedirs(item_dir, exist_ok=True)
        frame_creator = frame_creators[item_type]
        for idx in range(start, stop):
            rel_idx = idx - start
            frame = data_items[rel_idx]
            fig, ax = frame_creator(frame)
            meta_adder(fig, ax, item_type, metadata[rel_idx])
            eviz.save_figure(fig, os.path.join(item_dir, 'frame-{}'.format(idx)))
