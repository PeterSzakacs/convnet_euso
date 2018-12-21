import argparse
import sys

import cmdint.argparse_types as atypes
import cmdint.common_args as cargs
import utils.dataset_utils as ds

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

    dataset = ds.numpy_dataset.load_dataset(srcdir, name)
    dataset.shuffle_dataset(args.num_shuffles)
    dataset.save(srcdir)