import argparse
import sys

import utils.dataset_utils as ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Shuffle dataset a given number of times'))
    parser.add_argument('--name', required=True,
                        help=('name of the dataset, input, targets and meta file names are constructed from these.'))
    parser.add_argument('--srcdir', required=True,
                        help=('directory containing input data and target files.'))
    parser.add_argument('--num_shuffles', type=int, default=1,
                        help=('Number of times the generated data should be shuffled randomly after creation'))

    args = parser.parse_args(sys.argv[1:])
    if args.num_shuffles < 0:
        raise ValueError('Number of times the data is shuffled cannot be negative')

    dataset = ds.numpy_dataset.load_dataset(args.srcdir, args.name)
    dataset.shuffle_dataset(args.num_shuffles)
    dataset.save(args.srcdir)