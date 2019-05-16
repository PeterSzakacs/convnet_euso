import dataset.dataset_utils as ds
import dataset.io.fs_io as io_utils


def get_cast_func(name, fp_precision=None, nullable=False):
    if name == 'str':
        fn = str
    elif name == 'int':
        fn = int
    elif name == 'float':
        if fp_precision is None:
            fn = float
        else:
            fn = lambda val: round(float(val), fp_precision)
    else:
        raise ValueError('Unknown type name: {}'.format(name))
    if nullable:
        return lambda val: val if val == '' else fn(val)
    else:
        return fn


def main(**settings):
    srcdir, outdir = settings['srcdir'], settings['outdir']
    name, outname = args['name'], args['outname']
    if outname is None:
        outname = name
    if outdir is None:
        outdir = srcdir
    io_handler = io_utils.DatasetFsPersistencyHandler(load_dir=srcdir,
                                                      save_dir=outdir)
    items_slice = args['items_slice']
    old_dataset = io_handler.load_dataset(name)
    new_dataset = ds.NumpyDataset(outname, old_dataset.accepted_packet_shape,
                                  item_types=old_dataset.item_types,
                                  dtype=old_dataset.dtype)
    new_dataset.merge_with(old_dataset, items_slice)
    io_handler.save_dataset(new_dataset)


if __name__ == "__main__":
    import cmdint.cmd_interface_splitter as cmd
    import sys

    # command line parsing
    cmd_int = cmd.CmdInterface()
    args = cmd_int.get_cmd_args(sys.argv[1:])
    print(args)
