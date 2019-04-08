import argparse
import sys

import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Split off new dataset from range of input dataset items '
                     'or shrink original dataset'))
    in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
    out_aliases = {'dataset name': 'out_name', 'dataset directory': 'outdir'}
    dset_args = dargs.dataset_args(input_aliases=in_aliases,
                                   output_aliases=out_aliases)

    group = parser.add_argument_group(title='Input dataset settings')
    dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT)
    # slice of dataset items to use for evaluation
    group.add_argument('--start_item', default=0, type=int,
                       help=('Index of first dataset item to use for '
                             'evaluation.'))
    group.add_argument('--stop_item', default=None, type=int,
                       help=('Index of the dataset item after the last '
                             'item to use for evaluation.'))

    group = parser.add_argument_group(title='Output dataset settings')
    dset_args.add_dataset_arg_double(group, dargs. arg_type.OUTPUT,
                                     required=False)

    args = parser.parse_args(sys.argv[1:])
    name, srcdir = dset_args.get_dataset_double(args, dargs.arg_type.INPUT)
    outname, outdir = dset_args.get_dataset_double(args, dargs.arg_type.OUTPUT)
    items_slice = slice(args.start_item, args.stop_item)
    if outname is None:
        outname = name
    if outdir is None:
        outdir = srcdir

    io_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir,
                                                         save_dir=outdir)
    old_dataset = io_handler.load_dataset(name)
    new_dataset = ds.numpy_dataset(outname, old_dataset.accepted_packet_shape,
                                   item_types=old_dataset.item_types,
                                   dtype=old_dataset.dtype)
    old_dataset = new_dataset.merge_with(old_dataset, items_slice)
    io_handler.save_dataset(new_dataset)
