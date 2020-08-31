import abc
import datetime
import os
import shutil
import unittest

import dataset.io.fs.facades as facades

# set to any "truthy" value, e.g. 1, True to enable slow (i.e. IO-bound) tests
RUN_SLOW_TESTS = os.getenv('RUN_SLOW_TESTS', 'false')
RUN_SLOW_TESTS = RUN_SLOW_TESTS.lower() in ['true', '1', 't', 'y', 'yes',
                                            'yeah', 'yup']

# directory used for IO performed in these tests
TEST_DIR = os.getenv('TEST_DIR', default=os.curdir)


@unittest.skipIf(not RUN_SLOW_TESTS, 'RUN_SLOW_TESTS is falsy, skipped')
class BaseFacadeTest(unittest.TestCase):

    _temp_dir = None

    @classmethod
    def setUpClass(cls):
        facade_key = cls._get_facade_key()
        assert facade_key is not None
        temp_dir = os.path.join(
            TEST_DIR, cls.get_temp_dir_id(facade_key)
        )
        os.makedirs(temp_dir, exist_ok=True)
        cls._facade = facades.FACADES[facade_key]
        cls._temp_dir = temp_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._temp_dir)

    @classmethod
    def get_temp_dir_id(cls, facade_key):
        return f"tmp_test_{facade_key}_{cls.__name__}"\
               f"_{int(datetime.datetime.now().timestamp())}"

    @classmethod
    @abc.abstractmethod
    def _get_facade_key(cls):
        pass
