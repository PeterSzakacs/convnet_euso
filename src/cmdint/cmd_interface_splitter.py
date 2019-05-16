import argparse

import cmdint.common.dataset_args as dargs

class CmdInterface():

    def __init__(self):
        parser = argparse.ArgumentParser(
            description=('Split off new dataset from range of input dataset items '
                         'or shrink original dataset'))
        in_aliases = {'dataset name': 'name', 'dataset directory': 'srcdir'}
        out_aliases = {'dataset name': 'out_name', 'dataset directory': 'outdir'}
        dset_args = dargs.DatasetArgs(input_aliases=in_aliases,
                                      output_aliases=out_aliases)

        group = parser.add_argument_group(title='Input dataset settings')
        dset_args.add_dataset_arg_double(group, dargs.arg_type.INPUT)
        # slice of dataset items to split off
        group.add_argument('--start_item', default=0, type=int,
                           help=('Index of first dataset item to use for '
                                 'evaluation.'))
        group.add_argument('--stop_item', default=None, type=int,
                           help=('Index of the dataset item after the last '
                                 'item to use for evaluation.'))

        group = parser.add_argument_group(title='Output dataset settings')
        dset_args.add_dataset_arg_double(group, dargs.arg_type.OUTPUT,
                                         required=False)
        self._parser = parser
        self._dset_args = dset_args

    def get_cmd_args(self, argsToParse):
        _args = self._parser.parse_args(argsToParse)

        args = {}
        args['name'], args['srcdir'] = self._dset_args.get_dataset_double(
            _args, dargs.arg_type.INPUT)
        args['outname'], args['outdir'] = self._dset_args.get_dataset_double(
            _args, dargs.arg_type.OUTPUT)
        args['item_slice'] = slice(_args.start_item, _args.stop_item)

        return args