import utils.io_utils as io_utils
import utils.dataset_utils as ds

def check_dataset_compatibility(attrs1, attrs2):
    return (attrs1['packet_shape'] == attrs2['packet_shape'] and 
            attrs1['item_types'] == attrs2['item_types'])

if __name__ == "__main__":
    import cmdint.cmd_interface_merger as cmd
    import sys

    # command line parsing
    cmd_int = cmd.cmd_interface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)

    # first pass: ensure all datasets are compatible with each other before 
    # loading their contents into memory or merging them
    in_dsets = args.dataset
    persistency_handlers = {}
    name, srcdir = in_dsets[0][:] 
    handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
    first_dataset = handler.load_empty_dataset(name)
    persistency_handlers[name] = handler
    for dset in in_dsets[1:]:
        name, srcdir = dset[:]
        handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir)
        dataset = handler.load_empty_dataset(name)
        if first_dataset.is_compatible_with(dataset):
            persistency_handlers[name] = handler
        else:
            raise ValueError('Incompatible datasets:\n {} (attrs: {})\n'
                             ' and {} (attrs: {}):'.format(first_dataset, 
                             dataset))
    # second pass: iteratively merge items from all datasets into the first
    # one and then save in the new directory
    outname, outdir = args.name, args.outdir
    first_dataset.name = outname
    output_handler = io_utils.dataset_fs_persistency_handler(save_dir=outdir)
    for name, handler in persistency_handlers.items():
        dataset = handler.load_dataset(name)
        first_dataset.merge_with(dataset)
    output_handler.save_dataset(first_dataset)