import datetime
import os
import shutil
import unittest

# set to any "truthy" value, e.g. 1, True to enable slow (i.e. IO-bound) tests
RUN_SLOW_TESTS = os.getenv('RUN_SLOW_TESTS', 'false')
RUN_SLOW_TESTS = RUN_SLOW_TESTS.lower() in ['true', '1', 't', 'y', 'yes',
                                            'yeah', 'yup']

# directory used for IO performed in these tests
TEST_DIR = os.getenv('TEST_DIR', default=os.curdir)


@unittest.skipIf(not RUN_SLOW_TESTS, 'RUN_SLOW_TESTS is falsy, skipped')
class BaseIOTest(unittest.TestCase):

    _temp_dir = None

    @classmethod
    def setUpClass(cls):
        dir_name = cls._get_test_dir_name()
        temp_dir = os.path.join(
            TEST_DIR, dir_name
        )
        os.makedirs(temp_dir, exist_ok=True)
        cls._temp_dir = temp_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._temp_dir)

    @classmethod
    def _get_test_dir_name(cls):
        dt = int(datetime.datetime.now().timestamp())
        return f"tmp_test_{cls.__name__}_{dt}"
