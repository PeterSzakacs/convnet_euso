import unittest

import test.test_setups as testset
import utils.target_utils as targ


class TestModuleFunctions(unittest.TestCase):

    def test_get_target_name_shower(self):
        target = targ.CLASSIFICATION_TARGETS['shower']
        name = targ.get_target_name(target)
        self.assertEqual(name, 'shower')

    def test_get_target_name_noise(self):
        target = targ.CLASSIFICATION_TARGETS['noise']
        name = targ.get_target_name(target)
        self.assertEqual(name, 'noise')

    def test_get_target_probabilities(self):
        output = [0, 0]
        output[targ.CLASSIFICATION_TARGETS['shower'].index(1)] = 0.2314123
        output[targ.CLASSIFICATION_TARGETS['noise'].index(1)] = 0.2321131
        probs = targ.get_target_probabilities(output, precision=3)
        self.assertEqual(probs['shower'], 0.231)
        self.assertEqual(probs['noise'], 0.232)


class TestTargetHolder(testset.DatasetTargetsMixin, unittest.TestCase):

    def test_get_targets_as_dict_empty(self):
        holder = targ.TargetsHolder()
        self.assertDictEqual(holder.get_targets_as_dict(),
                             {'classification': [], })

    def test_get_targets_as_dict_after_extend(self):
        holder = targ.TargetsHolder()
        targets = {'classification': self.mock_targets}
        holder.extend(targets)
        self.assertDictEqual(holder.get_targets_as_dict(),
                             {'classification': targets['classification']})

    def test_get_targets_as_dict_after_append(self):
        holder = targ.TargetsHolder()
        target = {'classification': self.mock_targets[0]}
        holder.append(target)
        self.assertDictEqual(holder.get_targets_as_dict(),
                             {'classification': [target['classification'], ]})

    def test_get_targets_as_arraylike_empty(self):
        holder = targ.TargetsHolder()
        self.assertTupleEqual(holder.get_targets_as_arraylike(), ([], ))

    def test_get_targets_as_arraylike_after_extend(self):
        holder = targ.TargetsHolder()
        targets = {'classification': self.mock_targets}
        holder.extend(targets)
        self.assertTupleEqual(holder.get_targets_as_arraylike(),
                              (targets['classification'], ))

    def test_get_targets_as_arraylike_after_append(self):
        holder = targ.TargetsHolder()
        target = {'classification': self.mock_targets[0]}
        holder.append(target)
        self.assertTupleEqual(holder.get_targets_as_arraylike(),
                              ([target['classification']], ))

    def test_shuffle(self):
        holder = targ.TargetsHolder()
        targets = {'classification': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        holder.extend(targets)
        exp_targets = targets.copy()
        def shuffler(seq):
            temp = seq[0]
            seq[0] = seq[1]
            seq[1] = temp
        shuffler(exp_targets['classification'])
        holder.shuffle(shuffler, lambda: None)
        self.assertDictEqual(holder.get_targets_as_dict(), exp_targets)

if __name__ == '__main__':
    unittest.main()
