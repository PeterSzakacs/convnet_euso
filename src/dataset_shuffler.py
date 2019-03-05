import argparse
import sys

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs
import dataset.io.fs_io as io_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Randomly shuffle dataset a given number of times'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
    dset_args = cargs.dataset_args(input_aliases=in_aliases)
    dset_args.add_dataset_arg_double(parser, cargs.arg_type.INPUT)
    parser.add_argument('--num_shuffles', type=atypes.int_range(0), default=0,
                        help='Number of times the dataset should be shuffled.')

    args = parser.parse_args(sys.argv[1:])
    name, srcdir = dset_args.get_dataset_double(args,cargs.arg_type.INPUT)
    io_handler = io_utils.dataset_fs_persistency_handler(load_dir=srcdir,
                                                         save_dir=srcdir)
    dataset = io_handler.load_dataset(name)
    dataset.shuffle_dataset(args.num_shuffles)
    io_handler.save_dataset(dataset)