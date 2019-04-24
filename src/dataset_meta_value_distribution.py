import argparse
import sys

import pandas as pd

import cmdint.common.argparse_types as atypes
import cmdint.common.dataset_args as dargs
import dataset.io.fs_io as io_utils
import utils.common_utils as cutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Get attribute value distribution of metadata attribute '
                     'within a dataset or subset of it'))
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help=('name of output file to write to. If not '
                              'provided, output to stdout.'))

    group = parser.add_argument_group(title='Dataset settings')
    in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
    dset_args = dargs.dataset_args(input_aliases=in_aliases)
    dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT)
    # slice of dataset items to use for evaluation
    group.add_argument('--start_item', default=0, type=int,
                       help=('Index of first dataset item to use for '
                             'evaluation.'))
    group.add_argument('--stop_item', default=None, type=int,
                       help=('Index of the dataset item after the last '
                             'item to use for evaluation.'))

    group = parser.add_argument_group(title='Attribute settings')
    group.add_argument('--attribute', required=True,
                       help='Metadata attribute whose value distribution '
                            'to check.')
    group.add_argument('--attribute_type', choices=('str', 'float', 'int'),
                       default='str',
                       help='Type of data in the attribute, default: string')
    group.add_argument('--fp_precision', type=atypes.int_range(0),
                       help='Number of decimal points to round floating-point '
                            'attributes to, default: 2')
    group.add_argument('--nullable', default=False, action='store_true',
                       help='If set, script will assume this attribute can '
                            'can be unset or set to None')

    args = parser.parse_args(sys.argv[1:])
    outfile = args.outfile
    attr = args.attribute
    name, srcdir = dset_args.get_dataset_double(args,dargs.arg_type.INPUT)
    io_handler = io_utils.DatasetFsPersistencyHandler(load_dir=srcdir)
    dset = io_handler.load_dataset(name)
    items_slice = slice(args.start_item, args.stop_item)
    metadata = dset.get_metadata(items_slice)

    attr_ype = args.attribute_type,
    precision, nullable = args.fp_precision, args.nullable
    cast_fn = cutils.get_cast_func(args.attribute_type, fp_precision=precision,
                                   nullable=nullable)
    for item in metadata:
        item[attr] = cast_fn(item[attr])

    meta_groups = pd.DataFrame(metadata).groupby(attr)
    outfile.write('{}\tnum_entries\r\n'.format(attr))
    for group_val, rows in meta_groups:
        outfile.write('{}\t{}\r\n'.format(group_val, len(rows)))
    outfile.close()
