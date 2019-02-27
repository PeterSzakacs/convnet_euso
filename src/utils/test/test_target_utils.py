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

    # helper methods (targets setup)

    def _create_targets(self, itm_slice):
        return {'classification': self.mock_targets[itm_slice]}

    # helper methods (custom asserts)

    def _assertTargetsArraylikeEqual(self, targets, exp_targets):
        self.assertTupleEqual(targets, exp_targets)

    def _assertTargetsDictEqual(self, targets, exp_targets):
        self.assertDictEqual(targets, exp_targets)

    # test setup

    @classmethod
    def setUpClass(cls):
        super(TestTargetHolder, cls).setUpClass(num_items=4)

    # test methods
    ## test item retrieval methods

    def test_get_targets_as_dict_empty(self):
        holder = targ.TargetsHolder()
        self._assertTargetsDictEqual(holder.get_targets_as_dict(),
                                     {'classification': [], })

    def test_get_targets_as_dict_with_items(self):
        targets = self._create_targets(slice(0, 2))
        holder = targ.TargetsHolder()
        holder.extend(targets)
        self._assertTargetsDictEqual(holder.get_targets_as_dict(), targets)

    def test_get_targets_as_arraylike_empty(self):
        holder = targ.TargetsHolder()
        self._assertTargetsArraylikeEqual(holder.get_targets_as_arraylike(),
                                          ([], ))

    def test_get_targets_as_arraylike_with_items(self):
        targets = self._create_targets(slice(0, 2))
        exp_targets = (targets['classification'], )
        holder = targ.TargetsHolder()
        holder.extend(targets)
        self._assertTargetsArraylikeEqual(holder.get_targets_as_arraylike(),
                                          exp_targets)

    ## test item addition methods

    def test_extend(self):
        targets = self._create_targets(slice(None))
        exp_targets = {'classification': targets['classification']}
        holder = targ.TargetsHolder()
        holder.extend(targets)
        self._assertTargetsDictEqual(holder.get_targets_as_dict(), exp_targets)

    def test_append(self):
        target = self._create_targets(0)
        exp_target = {'classification': [target['classification']]}
        holder = targ.TargetsHolder()
        holder.append(target)
        self._assertTargetsDictEqual(holder.get_targets_as_dict(), exp_target)

    ## test shuffle

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
        self._assertTargetsDictEqual(holder.get_targets_as_dict(), exp_targets)

if __name__ == '__main__':
    unittest.main()
