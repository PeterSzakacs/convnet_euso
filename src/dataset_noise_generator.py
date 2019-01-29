import random as rand

import numpy as np

import utils.common_utils as cutils
import utils.dataset_utils as ds
import utils.io_utils as io_utils
import utils.target_utils as targ


class bg_lambda_default_generator:

    def __init__(self, bg_lambda_range, rng_seed=None, lambda_precision=4):
        bg_lambda = cutils.check_and_convert_value_to_tuple(
            bg_lambda_range, 'bg_lambda')
        cutils.check_interval_tuple(bg_lambda, 'bg_lambda', lower_limit=0)
        self._lam_min, self._lam_max = bg_lambda_range[0:2]
        rng = rand.Random()
        rng.seed(rng_seed)
        self._rng = rng
        self.lambda_rounding_precision = lambda_precision

    @property
    def lambda_rounding_precision(self):
        return self._ndigits

    @lambda_rounding_precision.setter
    def lambda_rounding_precision(self, value=4):
        self._ndigits = int(value)

    def set_rng_seed(self, seed=None):
        self._rng.seed(seed)

    def __call__(self):
        return round(self._rng.uniform(self._lam_min, self._lam_max), 
                     self._ndigits)


class poisson_packet_generator:

    def __init__(self, packet_shape, lambda_generator):
        self.packet_shape = packet_shape
        self.lambda_generator = lambda_generator

    def __call__(self):
        lam = self.lambda_generator()
        packet = np.random.poisson(lam=lam, size=self.packet_shape)
        meta = {'bg_lambda': lam}
        return packet, meta


def main(**kwargs):
    bg_lambda_range = kwargs['bg_lambda']
    seed, precision = kwargs['seed'], kwargs['precision']
    lambda_gen = bg_lambda_default_generator(bg_lambda_range, rng_seed=seed, 
                                             lambda_precision=precision)

    packet_shape = kwargs['packet_shape']
    packet_generator = poisson_packet_generator(packet_shape, lambda_gen)

    item_types = kwargs['item_types']
    name, outdir = kwargs['name'], kwargs['outdir']
    num_items, dtype = kwargs['num_items'], kwargs['dtype']
    dataset = ds.numpy_dataset(name, packet_shape, item_types=item_types, 
                               dtype=dtype)
    handler = io_utils.dataset_fs_persistency_handler(save_dir=outdir)

    target = targ.get_target_name('noise')
    for idx in range(num_items):
        packet, meta = packet_generator()
        dataset.add_data_item(packet, target, meta)
    handler.save_dataset(dataset)


if __name__ == '__main__':
    import sys
    import cmdint.cmd_interface_noisegen as cmd

    # command line parsing
    ui = cmd.cmd_interface()
    args_dict = ui.get_cmd_args(sys.argv[1:])
    main(**args_dict)
