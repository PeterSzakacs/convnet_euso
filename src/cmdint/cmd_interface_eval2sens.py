import os
import argparse

import cmdint.common.argparse_types as atypes
import dataset.constants as cons

class cmd_interface():

    def __init__(self):
        parser = argparse.ArgumentParser(
        description=('Evaluate model classification sensitivity as a function '
                     'of given attributes from provided evaluation results in '
                     'TSV format.'))

        # input tsv
        parser.add_argument('infiles', nargs='+', metavar='INFILE',
                            help=('Names of input TSVs to read from. If not '
                                  'provided, read from stdin.'))

        # input settings
        group = parser.add_argument_group(title="Evaluation settings")
        group.add_argument('--class_target', required=True,
                           choices=cons.CLASSIFICATION_TARGETS,
                           help='Classification target for which to evaluate '
                                'sensitivity metrics.')
        group.add_argument('--column', required=True,
                           help='Name of attribute from TSV file to use for '
                                'evaluating classification sensitivity.')
        group.add_argument('--column_type', choices=('str', 'float', 'int'),
                           default='str',
                           help='Type of data in the column, default: string.')
        group.add_argument('--all_targets', nargs='+',
                           default=list(cons.CLASSIFICATION_TARGETS.keys()),
                           help='All classification targets present in the '
                                'evaluation results.')

        # output settings
        group = parser.add_argument_group(title="Output settings")
        group.add_argument('--outfile', required=True,
                           help=('Output filename (minus extension).'))
        group.add_argument('--xscale', choices=('linear', 'log'),
                           default='linear',
                           help='Scale of x-axis in generated plot to use '
                                '(linear or logarithmic, default: linear).')
        group.add_argument('--plot_yerr', action='store_true',
                           help='Plot sensitivity error.')
        group.add_argument('--plot_colors', nargs='*', metavar='COLOR',
                           help='Color of each line in the plot. Must be same '
                                'length as list of infiles if provided.')
        group.add_argument('--plot_labels', nargs='*', metavar='LABEL',
                           help='Label of each line in the plot. Must be same '
                                'length as list of infiles if provided.')
        group.add_argument('--legend_fontsize', type=atypes.int_range(1),
                           help='Font size of plot legend.')
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

        exp_len = len(args.infiles)
        plot_settings_args = ('plot_colors', 'plot_labels', )
        for argname in plot_settings_args:
            argval = getattr(args, argname)
            if argval is not None and len(argval) != exp_len:
                raise ValueError('Invalid number of entries for "{}", '
                                 'expected {}'.format(argname, exp_len))

        arg_names = ('infiles', 'outfile', 'xscale', 'column', 'class_target',
                     'all_targets', 'plot_yerr', 'legend_fontsize',
                     *plot_settings_args)
        args_dict = {name: getattr(args, name) for name in arg_names}
        args_dict['column_type'] = self._get_column_cast_fn(args.column_type)

        return args_dict
