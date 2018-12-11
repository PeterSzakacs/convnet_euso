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

    dataset_dicts = []
    name, srcdir = args.datasets[0][:]
    dataset_dicts.append(ds.numpy_dataset.preload_dataset(srcdir, name))
    capacity = num_data = dataset_dicts[0]['num_data']
    for dataset in args.datasets[1:]:
        name, srcdir = dataset[:]
        attrs = ds.numpy_dataset.preload_dataset(srcdir, name)
        if check_dataset_compatibility(dataset_dicts[0], attrs):
            dataset_dicts.append(attrs)
            capacity += attrs['num_data']
            num_data += attrs['num_data']
        else: 
            raise ValueError('Incompatible datasets:\n {} (attrs: {})\n'
                             ' and {} (attrs: {}):'.format(args.datasets[0][0], 
                             dataset_dicts[0], name, attrs))

    dataset = args.datasets[0]
    name, srcdir = dataset[:]
    new_dataset = ds.numpy_dataset.load_dataset(srcdir, name)
    for idx in range(len(args.datasets[1:])):
        dataset = args.datasets[idx]
        name, srcdir = dataset[:]
        dataset = ds.numpy_dataset.load_dataset(srcdir, name)
        new_dataset.merge_with(dataset)

    new_dataset.name = args.name
    new_dataset.save(args.outdir)


