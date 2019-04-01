import os
import argparse
import sys

import dataset.constants as cons

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
        description=('Evaluate model classification sensitivity as a function '
                     'of given attributes from provided evaluation results in '
                     'TSV format.'))

        # input tsv
        parser.add_argument('infile',
                            help=('Name of input TSV to read from. If not '
                                  'provided, read from stdin.'))

        # output settings
        group = parser.add_argument_group(title="Output settings")
        group.add_argument('--logdir', default=os.path.curdir,
                           help=('Directory to store output logs. If a '
                                 'non-default directory is used, it must '
                                 'exist prior to calling this script.'))
        group.add_argument('--xscale', choices=('linear', 'log'),
                           default='linear',
                           help='Scale of x-axis in generated plot to use '
                                '(linear or logarithmic)')
        group.add_argument('--column', required=True,
                           help='Name of attribute from TSV file to use for '
                                'evaluating classification sensitivity.')
        group.add_argument('--column_type', choices=('str', 'float', 'int'),
                           default='str',
                           help='Type of data in the column, default: string')
        group.add_argument('--class_target', required=True,
                           choices=cons.CLASSIFICATION_TARGETS,
                           help='Classification target for which to evaluate '
                                'sensitivity metrics')
        group.add_argument('--all_targets', nargs='+',
                           default=list(cons.CLASSIFICATION_TARGETS.keys()),
                           help='All classification targets present in the '
                                'evaluation results')
        self.parser = parser

    def _get_column_cast_fn(self, column_type):
        if column_type == 'str':
            return lambda val: val
        elif column_type == 'float':
            return float
        elif column_type == 'int':
            return int
        else:
            raise Exception('Unknown column type {}'.format(column_type))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        arg_names = ['infile', 'logdir', 'xscale', 'column', 'class_target',
                     'all_targets']
        args_dict = {name: getattr(args, name) for name in arg_names}
        args_dict['column_type'] = self._get_column_cast_fn(args.column_type)

        return args_dict
