import inspect
import typing as t


T = t.TypeVar('T')


class ClassInstanceProvider:
    """
    Factory/Provider which allows constructing objects using special keys to
    identify the specific class whose instance it should create.

    This class has a similar function to custom dependency providers in the
    Angular framework (hence the name) or a factory class for creating bean
    instances in Java EE, since its primary intended use is for runtime
    resolution of dependencies based on simple keywords.

    Currently only classes without any mandatory initializer/__init__() method
    params are supported (support to be added later).

    Features/highlights:
    - keys to which a given class is bound are case-insensitive
    - ability to set the same class for multiple different keys
    - lazy instance creation and caching of already created instances for
      the same key (default behavior, which can be turned off per call)
    - ability to set a base class on initialization - if set, any classes
      that are set need to be subclasses of it
    - ability to pre-set the cached instance when setting a class for a
      given key
    """

    def __init__(
            self,
            class_map: t.Mapping[str, t.Mapping[str, t.Any]] = None,
            base_class: t.Type = None,
    ):
        """Construct a new provider.

        The classes for which this provider instance should be responsible can
        optionally be initialized here using the class_map parameter. Every key
        in this mapping is bound to a value which is itself a mapping with the
        following structure:

        'type_key': {
            'class': <class/type object, e.g. SomeClass>,
            'inst': (optional) <pre-cached instance of class/type>
        }

        :param class_map: Mapping of keywords to use class/type objects
        :param base_class: (Optional) class which must be the superclass of all
                           provided classes that are set for this instance
        """
        # create a deep copy of the passed in class map
        self._base = base_class
        self._classes = {}
        self._cache = {}
        class_map = class_map or {}
        for class_key, class_dict in class_map.items():
            self.set_class(class_key, class_dict['class'],
                           cached_instance=class_dict.get('inst'))

    @property
    def base_class(self) -> t.Type:
        """Base class for all classes managed by this provider instance.

        :return: Base class or None if it was not set.
        """
        return self._base

    @property
    def available_keys(self) -> t.Set[str]:
        """Set of all configured keys representing the configured classes that
        are managed by this provider.

        :return: Set of all managed keys for this provider instance.
        """
        return set(self._classes.keys())

    def get_class(self, key: str) -> t.Type:
        """Retrieve the class/type bound to the given key (case-insensitive).

        :param key: The key to which this class is bound.
        :return: The class bound to the given key.
        """
        return self._classes.get(key.lower(), {'class': None})['class']

    def set_class(
            self,
            key: str,
            cls: t.Type[T],
            cached_instance: T = None,
    ) -> None:
        """Set class/type to provide instances of for the given key.

        :param key: key to bind to the given class/type
        :param cls: the class/type from which to create instances
        :param cached_instance: (optional) an already created instance of the
                                class which will be cached. Note that for
                                consistent behavior, subclass instances are NOT
                                permitted.
        """
        _key = key.lower()
        _base = self._base
        if _base and not issubclass(cls, _base):
            _exp_base = self._format_classname(_base)
            _cls = self._format_classname(cls)
            raise ValueError(f'Upper type bound \'{_exp_base}\' not satisfied '
                             f'by type \'{_cls}\'')
        if inspect.isabstract(cls):
            raise ValueError('Abstract classes are not permitted')
        if cached_instance and not type(cached_instance) == cls:
            _exp_cls = self._format_classname(cls)
            _inst_cls = self._format_classname(type(cached_instance))
            raise ValueError(f'Mismatch of instance type \'{_inst_cls}\' and '
                             f'class \'{_exp_cls}\' for key \'{_key}\'')
        self._classes[_key] = {
            'class': cls,
        }
        if cached_instance:
            self._cache[_key] = cached_instance

    def get_instance(
            self,
            key: str,
            new_instance: bool = False,
    ):
        """Create instance from the class bound to the given key.

        This method by default returns a cached shared instance (singleton)
        if it exists, unless new_instance is set to True.

        Do note that if the cache is empty during a call and new_instance=True,
        the created instance is NOT cached for later calls. We implicitly
        assume the caller does not want such instance to be potentially shared
        with any other code.

        :param key: key that is bound to the given class/type
        :param new_instance: boolean flag, if set to True, a new instance is
                             guaranteed to be created (bypassing the cache)
        :return: object instance of the type/class bound to the given key
        """
        _key = key.lower()
        if _key not in self._classes:
            raise ValueError(f'No class set for key \'{key}\'')

        # if user does not require a new instance, fetch it from the cache
        _new_inst = bool(new_instance)
        _inst = self._cache.get(_key)
        if _inst is not None and not _new_inst:
            return _inst

        # instantiate and optionally cache the new instance
        _class_spec = self._classes[_key]
        _cls = _class_spec['class']
        _inst = _cls()

        if not _new_inst:
            self._cache[_key] = _inst

        return _inst

    @staticmethod
    def _format_classname(cls: t.Type):
        return f"{cls.__module__}.{cls.__name__}"
