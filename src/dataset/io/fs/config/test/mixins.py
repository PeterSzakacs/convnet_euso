import random
import typing as t
import uuid

import numpy as np

import dataset.io.fs.facades as facades
import dataset.io.fs.utils as utils


class DatasetConfigRandomGeneratorTestMixin:

    def _generate_section_config(
            self,
            add_num_items=True,
            **kwargs
    ) -> t.MutableMapping[str, t.Any]:
        """Return simulated dataset section config.

        Recognized parameters from kwargs correspond to the top-level dataset
        section keys and always override the generated defaults:

        - num_items
        - backend
        - types

        :param kwargs: Overrides for the top level keys of the dataset section
                       config.
        :return: A simulated dataset section config.
        """
        randint = random.randint
        backend = kwargs.get('backend') or self._generate_backend_config()
        types = kwargs.get('types') or self._generate_types_config()
        config = {
            'backend': backend, 'types': types,
        }
        if add_num_items:
            config['num_items'] = (kwargs.get('num_items') or randint(1, 1000))
        return config

    def _generate_backend_config(
            self,
            backend_names: t.Sequence[str] = None,
            filename_formats: t.Sequence[str] = None,
    ) -> t.MutableMapping[str, t.Any]:
        """Generate simulated backend config as present in a dataset section
        config.

        :param backend_names: (optional) List of allowed storage backends from
                              which to pick a random value.
        :param filename_formats: (optional) List of defined filename formats
                                 from which to pick a random value.
        :return:
        """
        choice = random.choice
        backend_names = backend_names or list(
            facades.get_facades_provider().available_keys
        )
        filename_formats = filename_formats or list(
            utils.get_formatters_provider().available_keys
        )

        filename_extensions = []
        filename_extensions.extend(backend_names)
        filename_extensions.extend(str(uuid.uuid4()) for idx in range(3))

        return {
            'name': choice(backend_names),
            'filename_extension': choice(filename_extensions),
            'filename_format': choice(filename_formats),
        }

    def _generate_types_config(
            self,
            item_types: t.Sequence[str] = None,
            allowed_item_types: t.Sequence[str] = None,
    ) -> t.MutableMapping[str, t.MutableMapping[str, t.Any]]:
        """Generate simulated item type config mapping as present in a dataset
        section config.

        The exact item types for which type configs are to be generated are
        resolved using the following logic:

        - if 'item_types' is defined and not empty, this method only generates
          configs for all types in the sequence
        - if 'item_types' is None or empty, but 'allowed_item_types' is a
          non-empty sequence, this method selects a sample of random size of
          item types from this sequence for which it generates configs
        - if both parameters are false, we generate a random sized sequence of
          item types for which we generate type configs

        :param item_types: (optional) Sequence of item types for which to
                           generate type configs.
        :param allowed_item_types: (optional) Sequence of item_types from which
                                   a random sample is selected to generate
                                   configs for.
        :return: The item types config as a mapping from item type identifier
                 to the config for the item type
        """
        randint = random.randint
        choice = random.choice
        sample = random.sample

        if item_types is None:
            if allowed_item_types is None:
                allowed_item_types = [
                    str(uuid.uuid4()) for idx in range(randint(4, 10))
                ]
            item_types = sample(
                allowed_item_types, randint(1, len(allowed_item_types))
            )

        item_dtypes = self._get_item_dtypes()
        item_shapes = self._generate_item_shapes(item_types)

        return {item_type: {'dtype': choice(item_dtypes),
                            'shape': item_shapes[item_type]}
                for item_type in item_types}

    def _generate_item_shapes(
            self,
            item_types: t.Sequence[str],
    ) -> t.MutableMapping[str, t.Tuple[int]]:
        """Generate shapes for the passed in item types.

        :param item_types: The item types for which to generate shape.
        :return: A mapping from item type to its shape as a tuple of ints.
        """
        randint = random.randint
        return {
            item_type: (randint(2, 20), randint(4, 40))
            for item_type in item_types
        }

    def _create_items_from_config(
            self,
            config: t.Mapping[str, t.Any],
    ) -> t.MutableMapping[str, np.ndarray]:
        """Generate ndarrays representing dataset items from the passed section
        config.

        The returned ndarrays for each corresponding item type config as found
        in the config['types'] mapping will be created with the same dtype as
        in the config and a shape expressed as the tuple:

        (config['num_items'], *config['types'][item_type]['shape'])

        :param config: Mapping representing the dataset section config.
        :return: A mapping from item type to ndarray matching the corresponding
                 item type config specification.
        """
        num_items, types_config = config['num_items'], config['types']
        return {
            item_type: np.empty(
                shape=(num_items, *item['shape']), dtype=item['dtype']
            ) for item_type, item in types_config.items()
        }

    def _get_item_dtypes(
            self,
    ) -> t.Sequence:
        sctypes = np.sctypes
        return [
            sctype.__name__ for sctype in
            (sctypes['int'] + sctypes['uint'] + sctypes['float'])
        ]
