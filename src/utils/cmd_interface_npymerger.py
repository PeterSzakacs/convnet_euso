import os
import argparse

class cmd_interface():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Merge multiple npy files with frame packets into a single dataset to use for training neural networks')
        self.parser.add_argument('-s', '--srcdir', required=True,
                                help=('name of the directory root from which to retrieve source npy files'))
        self.parser.add_argument('-i', '--infile', required=True,
                                help=('name of the tsv file storing the list of npy files under srcdir to use'
                                     ' for generating the result file'))
        self.parser.add_argument('-o', '--outfile', required=True,
                                help=('name of the output npy file (sans .npy extension)'))
        self.parser.add_argument('-t', '--targetfile', required=True,
                                help=('name of the output targets npy file (sans .npy extension)'))

    def get_cmd_args(self, argsToParse):
        args = self.parser.parse_args(argsToParse)

        return args

