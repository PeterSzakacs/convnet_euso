import abc
import random
import unittest

from .. import test_utils as tu
from .. import providers


class MockBaseClass(abc.ABC):

    @abc.abstractmethod
    def foo(self):
        pass


class A(MockBaseClass):
    def foo(self):
        pass


class B(MockBaseClass):
    def foo(self):
        pass


class AExtended(A):
    pass


class SomeOtherBaseClass:
    pass


class TestClassInstanceProvider(unittest.TestCase):

    @classmethod
    def _get_provider(cls,
                      class_map=None,
                      base_class=None):
        return providers.ClassInstanceProvider(
            class_map=class_map, base_class=base_class
        )

    @classmethod
    def _get_class_map(cls):
        return {
            'a_class_key': {
                'class': A
            },
            'b_class_key': {
                'class': B
            },
        }

    @classmethod
    def _get_base_class(cls):
        return MockBaseClass

    # property tests

    def test_get_available_keys(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        exp_keys = set(class_map.keys())
        provider = provider or self._get_provider(class_map=class_map)
        self.assertSetEqual(provider.available_keys, exp_keys)

    def test_get_base_class(
            self, provider=None, base_class=None
    ):
        base_class = base_class or self._get_base_class()
        provider = provider or self._get_provider(base_class=base_class)
        self.assertEqual(provider.base_class, base_class)

    def test_get_class(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        # pick a random entry from the class map
        key = random.choice(list(class_map))
        exp_class = class_map[key]['class']

        self.assertEqual(provider.get_class(key), exp_class)

    # get_class() tests

    def test_get_class_for_non_existing_key(
            self, provider=None, class_map=None, key='non_existing'
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        self.assertIsNone(provider.get_class(key))

    def test_key_is_case_insensitive_for_get_class(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        # pick a random entry from the class map
        key = random.choice(list(class_map))
        exp_class = class_map[key]['class']

        # the provider should return the correct class even if the passed in
        # key does not match in case
        actual_class = provider.get_class(key.upper())
        self.assertEqual(actual_class, exp_class)

    # get_instance() tests

    def test_key_is_case_insensitive_for_get_instance(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        # pick a random entry from the class map
        key = random.choice(list(class_map))
        exp_class = class_map[key]['class']

        # provider should create an instance of exp_class even if the passed in
        # key does not match in case
        inst = provider.get_instance(key.upper())
        self.assertIsInstance(inst, exp_class)

    def test_caching_for_get_instance(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)
        key = random.choice(list(class_map))

        inst1 = provider.get_instance(key)
        inst2 = provider.get_instance(key)

        # both references should point to the same object
        self.assertIs(inst1, inst2)

    def test_get_instance_with_new_instance_and_cache_already_set(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)
        key = random.choice(list(class_map))

        inst1 = provider.get_instance(key)
        inst2 = provider.get_instance(key, new_instance=True)

        # both references should point to different objects, i.e. the second
        # call should not return the cached instance
        self.assertIsNot(inst1, inst2)

    def test_get_instance_with_new_instance_and_without_cache_set(
            self, provider=None, class_map=None
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)
        key = random.choice(list(class_map))

        inst1 = provider.get_instance(key, new_instance=True)
        inst2 = provider.get_instance(key)

        # both references should point to different objects, i.e. the instance
        # created in the first call should not be cached
        self.assertIsNot(inst1, inst2)

        # safety check to make sure the actual assertion doesn't pass in case
        # the 2 objects are instances of different classes
        self.assertIsInstance(inst1, inst2.__class__)

    # set_class() tests

    def test_key_is_case_insensitive_for_set_class(
            self, provider=None, class_map=None, key='some_key', cls=AExtended
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)
        provider.set_class(key.upper(), cls)

        # the class should be retrievable by the original key even if the key
        # passed in set_class differs in case
        self.assertEqual(provider.get_class(key), cls)

        # also check that the available_keys property was updated
        keys = provider.available_keys
        exp_keys = set(class_map.keys()).union([key.lower()])
        self.assertSetEqual(keys, exp_keys)

    def test_set_class_with_new_key(
        self, provider=None, class_map=None, key='unique', cls=AExtended
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        provider.set_class(key, cls)

        # check the type of instance retrieved for the given key
        inst = provider.get_instance(tu.random_capitalize(key))
        self.assertIsInstance(inst, cls)

        # also check the available_keys property has been updated
        keys = provider.available_keys
        exp_keys = set(class_map.keys()).union([key])
        self.assertSetEqual(keys, exp_keys)

    def test_set_class_with_existing_key(
        self, provider=None, class_map=None, cls=AExtended
    ):
        class_map = class_map or self._get_class_map()
        provider = provider or self._get_provider(class_map=class_map)

        key = random.choice(list(class_map))
        provider.set_class(key, cls)

        # check the type of instance retrieved for the given key
        inst = provider.get_instance(tu.random_capitalize(key))
        self.assertIsInstance(inst, cls)

        # also check the available_keys property is still the same
        keys = provider.available_keys
        exp_keys = set(class_map.keys())
        self.assertSetEqual(keys, exp_keys)

    def test_set_class_with_abstract_class(
            self, provider=None, key='key', cls=MockBaseClass
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map()
        )
        # Also test the error message
        exp_msg = 'Abstract classes are not permitted'

        # ValueError makes the most sense as all args are of permitted types
        self.assertRaisesRegex(
            ValueError, exp_msg, provider.set_class,
            key, cls
        )

    def test_set_class_with_class_not_a_subtype_of_base_class(
            self, provider=None, key='key', cls=SomeOtherBaseClass
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map(), base_class=self._get_base_class()
        )
        base = provider.base_class

        # Also test the error message - should specify full path of both
        # classes, i.e. some.module.ClassName
        cls_desc = f"{cls.__module__}.{cls.__name__}"
        base_desc = f"{base.__module__}.{base.__name__}"
        exp_msg = (f'Upper type bound \'{base_desc}\' not satisfied '
                   f'by type \'{cls_desc}\'')

        # ValueError makes the most sense as all args are of permitted types
        self.assertRaisesRegex(
            ValueError, exp_msg, provider.set_class,
            key, cls
        )

    def test_set_class_with_cached_instance(
            self, provider=None, key='key', cls=AExtended, inst=AExtended()
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map()
        )

        provider.set_class(key, cls, cached_instance=inst)
        _inst = provider.get_instance(key)

        # provider should return inst if new_instance == False
        self.assertIs(_inst, inst)

    def test_set_class_with_cached_instance_of_wrong_class(
            self, provider=None, key='key', cls=A, inst=B()
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map()
        )
        self._negative_cached_instance_test(provider, key, cls, inst)

    def test_set_class_with_cached_instance_of_subtype(
            self, provider=None, key='key', cls=A, inst=AExtended()
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map()
        )
        self._negative_cached_instance_test(provider, key, cls, inst)

    def test_set_class_with_cached_instance_of_supertype(
            self, provider=None, key='key', cls=AExtended, inst=A()
    ):
        provider = provider or self._get_provider(
            class_map=self._get_class_map()
        )
        self._negative_cached_instance_test(provider, key, cls, inst)

    # helper methods:

    def _negative_cached_instance_test(self, provider, key, cls, inst):
        inst_cls = type(inst)

        # Also test the error message - should specify full path of both
        # classes, i.e. some.module.ClassName
        cls_desc = f"{cls.__module__}.{cls.__name__}"
        inst_desc = f"{inst_cls.__module__}.{inst_cls.__name__}"
        exp_msg = (f'Mismatch of instance type \'{inst_desc}\' and '
                   f'class \'{cls_desc}\' for key \'{key}\'')

        # ValueError makes the most sense as all args are of permitted types
        self.assertRaisesRegex(
            ValueError, exp_msg, provider.set_class,
            key, cls, cached_instance=inst
        )


if __name__ == '__main__':
    unittest.main()
