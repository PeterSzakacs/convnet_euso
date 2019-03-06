import argparse
import unittest

import cmdint.common.network_args as net_args


class TestNetworkArgs(unittest.TestCase):

    def test_add_network_arg_custom_arg_name(self):
        arg_name, value = 'net_test_name', 'somenet'
        cmdline = ['--{}'.format(arg_name), value]
        parser = argparse.ArgumentParser()
        net_args.add_network_arg(parser, arg_name=arg_name)
        res = parser.parse_args(cmdline)
        self.assertEqual(getattr(res, arg_name), value)

    def test_add_network_arg_short_alias(self):
        alias, value = 'n', 'some_test_net'
        cmdline = ['-{}'.format(alias), value]
        parser = argparse.ArgumentParser()
        net_args.add_network_arg(parser, short_alias=alias)
        res = parser.parse_args(cmdline)
        self.assertEqual(res.network, value)

    def test_network_arg_missing_when_required_raises_error(self):
        parser = argparse.ArgumentParser()
        net_args.add_network_arg(parser, required=True)
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args([])
            self.fail('Failed to raise exception on network not passed')


class TestModelFileArgs(unittest.TestCase):

    def test_add_model_file_arg_custom_arg_name(self):
        arg_name, value = 'model_file_test', 'some_model_file.dat'
        cmdline = ['--{}'.format(arg_name), value]
        parser = argparse.ArgumentParser()
        net_args.add_model_file_arg(parser, arg_name=arg_name)
        res = parser.parse_args(cmdline)
        self.assertEqual(getattr(res, arg_name), value)

    def test_add_model_file_arg_short_alias(self):
        alias, value = 'm', 'model_file.dat'
        cmdline = ['-{}'.format(alias), value]
        parser = argparse.ArgumentParser()
        net_args.add_model_file_arg(parser, short_alias=alias)
        res = parser.parse_args(cmdline)
        self.assertEqual(res.model_file, value)

    def test_model_arg_missing_when_required_raises_error(self):
        parser = argparse.ArgumentParser()
        net_args.add_model_file_arg(parser, required=True)
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args([])
            self.fail('Failed to raise exception on model_file not passed')


class TestTrainingSettingsArgs(unittest.TestCase):

    def test_custom_short_alias(self):
        custom = {'short_alias': 'o'}
        parser = argparse.ArgumentParser()
        net_args.add_training_settings_args(parser, optimizer=custom,
                                            num_epochs={'required': False})
        res = parser.parse_args(['-o', 'Adam'])
        self.assertEqual(getattr(res, 'optimizer'), 'Adam')

    def test_custom_default_value(self):
        custom = {'default': 0.01}
        parser = argparse.ArgumentParser()
        net_args.add_training_settings_args(parser, learning_rate=custom,
                                            excluded_args=['num_epochs'])
        res = parser.parse_args([])
        self.assertEqual(getattr(res, 'learning_rate'), 0.01)

    def test_custom_required_arg(self):
        custom = {'required': True}
        parser = argparse.ArgumentParser()
        net_args.add_training_settings_args(parser, loss_fn=custom,
                                            excluded_args=['num_epochs'])
        with self.assertRaises(SystemExit) as cm:
            args = parser.parse_args([])
            self.fail('Failed to raise exception on required arg not passed')

    def test_custom_type(self):
        custom = {'type': float}
        parser = argparse.ArgumentParser()
        net_args.add_training_settings_args(parser, num_epochs=custom)
        res = parser.parse_args(['--num_epochs', '0.01'])
        self.assertEqual(getattr(res, 'num_epochs'), 0.01)

    def test_excluded_args(self):
        excluded = set(net_args.TRAIN_SETTINGS_ARGS.keys())
        excluded.remove('num_epochs')
        parser = argparse.ArgumentParser()
        net_args.add_training_settings_args(parser, excluded_args=excluded)
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(['--num_epochs', '10', '--optimizer', 'Adam'])
            self.fail('Failed to raise exception on unknown arg passed')


if __name__ == '__main__':
    unittest.main()
