import abc

import dataset.io.fs.test.utils as test_utils
import dataset.io.fs.facades as facades


class BaseFacadeTest(test_utils.BaseIOTest):
    """Base class for tests which only test a single facade class identifiable
    via its key/name as used in the facades.FACADES dict
    """

    @classmethod
    def setUpClass(cls):
        super(BaseFacadeTest, cls).setUpClass()
        facade_key = cls._get_facade_key()
        provider = facades.get_facades_provider()
        cls._facade = provider.get_instance(
            facade_key, new_instance=True
        )

    @classmethod
    def _get_temp_dir_id(cls):
        # append facade name/key to the end of the test dir
        return f"{super(cls)._get_test_dir_name()}_{cls._get_facade_key()}"

    @classmethod
    @abc.abstractmethod
    def _get_facade_key(cls):
        pass
