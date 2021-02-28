import typing as t
import unittest.mock as mock

import numpy as np

import common.providers as providers
from .. import get_facades_provider


class IoFacadeMocksTestMixin:

    def _get_mock_facades(
            self,
            load_retval=None,
            facades_provider: providers.ClassInstanceProvider = None
    ) -> t.Mapping[str, mock.MagicMock]:
        """Create test mocks for all IO facades defined in 'facades_provider'
        and set the load method for all of them to return the value specified
        with 'load_retval'.

        If 'load_retval' is not defined, this method will simply set the load
        method to use a static ndarray filled with ones instead.

        If 'facades_provider' is not defined, this method will mock all facades
        from the provider returned by :func:`get_facades_provider`.

        The return value of this method can be directly passed to
        :method:`self._get_mock_facades_provider`.

        :param load_retval: (optional) The return value to assign for calls to
                            the 'load' method of the facade.
        :param facades_provider: (optional) The class instance provider with
                                 all defined facades to be mocked.
        :return: A mapping of facade keys available in the provider to mocks
                 of the real instances originally bound to those keys.
        """
        _retval = load_retval or np.ones(shape=(1, 10, 2))
        _provider = facades_provider or get_facades_provider()
        _available = _provider.available_keys
        _mocks = {key: mock.create_autospec(_provider.get_instance(key))
                  for key in _available}
        for key, facade in _mocks.items():
            facade.load.return_value = _retval
        return _mocks

    def _get_mock_facades_provider(
            self,
            mock_facades: t.Mapping[str, mock.MagicMock]
    ) -> providers.ClassInstanceProvider:
        """Convert the provided mapping of facade key to facade mock instances
        into a configured class instance provider returning the mock instances.

        :param mock_facades: Mapping of facade key to mock facade instance.
        :return: Class instance provider with all facades defined in the passed
                 mapping bound to their original keys.
        """
        _provider = providers.ClassInstanceProvider()
        for key, facade in mock_facades.items():
            _provider.set_class(key, type(facade), cached_instance=facade)
        return _provider
