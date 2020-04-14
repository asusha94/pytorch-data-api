import copy
import operator

_STRATEGIES = []


class _DefaultStrategy:
    @staticmethod
    def is_supported(item_type):
        return True

    def __init__(self, item, batch_size, padded_shape, padding_value):
        self._batch_size = batch_size
        self._padding_value = padding_value

        if isinstance(item, str):
            self._padding_value = str(self._padding_value) if self._padding_value is not None else '\0'
            self._can_be_padded = True
        elif isinstance(item, bytes):
            self._padding_value = bytes(self._padding_value) if self._padding_value is not None else b'\0'
            self._can_be_padded = True
        elif isinstance(item, list):
            self._padding_value = [self._padding_value]
            self._can_be_padded = True
        elif isinstance(item, tuple):
            self._padding_value = (self._padding_value,)
            self._can_be_padded = True

        if padded_shape is not None:
            if isinstance(padded_shape, (tuple, list)):
                self._padded_shape = padded_shape[0] if len(padded_shape) else None
            else:
                self._padded_shape = padded_shape
        else:
            self._padded_shape = None

    def make_batch(self, items, idx):
        if not self._can_be_padded:
            batch = [copy.deepcopy(item) for item in items]
        else:
            lens = [len(item) for item in items]
            max_len = max(lens)

            padded_len = self._padded_shape if self._padded_shape is not None else max_len

            batch = [self._pad_item(item[idx], padded_len) for item in items]

        if len(batch) < self._batch_size:
            batch = batch + [None] * (self._batch_size - len(batch))

        return batch

    def _pad_item(self, item, padded_len):
        if len(item) > padded_len:
            return copy.deepcopy(item[:padded_len])
        elif len(item) < padded_len:
            return item + self._padding_value * (padded_len - len(item))
        else:
            return copy.deepcopy(item)


_STRATEGIES.append(_DefaultStrategy)

try:
    import numpy as np

    class _NumpyStrategy:
        @staticmethod
        def is_supported(item_type):
            return item_type.__module__ == np.__name__ or item_type in [float, int]

        def __init__(self, item, batch_size, padded_shape, padding_value):
            if isinstance(item, np.ndarray):
                self._dtype = item.dtype
                self._ndim = item.ndim
            else:
                self._dtype = np.dtype(type(item))
                self._ndim = 0

            self._batch_size = batch_size
            self._padded_shape = padded_shape
            self._padding_value = padding_value

            if self._padded_shape is not None:
                self._padded_shape = [(None if s is not None and s < 1 else s) for s in self._padded_shape]
            else:
                self._padded_shape = [None] * self._ndim

            if self._padding_value is None:
                self._padding_value = 0

        def _batch_insert(self, batch, idx, item):
            if item is None:
                batch[idx] = 0
            else:
                max_shape = batch.shape[1:]

                inner_slices = (idx,) + tuple(slice(0, s, 1) for s in item.shape)
                outer_slices = (idx,) + tuple(slice(s, m, 1) for s, m in zip(item.shape, max_shape))
                all_slices = (idx,) + tuple(slice(0, m, 1) for m in max_shape)

                batch[inner_slices] = item
                for i in range(1, len(outer_slices)):
                    slices = all_slices[:i] + outer_slices[i:i + 1]
                    if i < len(outer_slices) - 1:
                        slices = slices + all_slices[i + 1:]

                    batch[slices] = self._padding_value

        def make_batch(self, items, idx):
            if self._ndim == 0:
                if self._padded_shape is not None:
                    batch = np.empty([self._batch_size] + self._padded_shape, dtype=self._dtype)

                    for i, item in enumerate(items):
                        item = item[idx]
                        item = np.full([1] * len(self._padded_shape), item, dtype=self._dtype)
                        self._batch_insert(batch, i, item)
                else:
                    batch = np.empty([self._batch_size], dtype=self._dtype)
                    for i, item in enumerate(items):
                        item = item[idx]
                        if item is None:
                            batch[i] = 0
                        else:
                            batch[i] = item[idx]
            else:
                shapes = [np.shape(s[idx]) for s in items]
                max_shape = [(max(shapes, key=operator.itemgetter(i))[i]
                              if self._padded_shape[i] is None
                              else self._padded_shape[i])
                             for i in range(self._ndim)]

                expand_dims = 0
                if len(max_shape) < len(self._padded_shape):
                    expand_dims = len(self._padded_shape) - len(max_shape)
                    max_shape = max_shape + [(self._padded_shape[i] if self._padded_shape[i] is not None else 1)
                                             for i in range(len(max_shape), len(self._padded_shape))]

                batch = np.empty([self._batch_size] + max_shape, dtype=self._dtype)

                for i, item in enumerate(items):
                    item = item[idx]
                    if expand_dims:
                        for _ in range(expand_dims):
                            item = np.expand_dims(item, axis=-1)

                    self._batch_insert(batch, i, item)

            if len(items) < self._batch_size:
                batch[len(items):] = 0

            return batch

    _STRATEGIES.insert(0, _NumpyStrategy)
except (ImportError, ModuleNotFoundError):
    pass

try:
    import torch

    class _TorchStrategy:
        @staticmethod
        def is_supported(item_type):
            return item_type == torch.Tensor

        def __init__(self, item, batch_size, padded_shape, padding_value):
            self._dtype = item.dtype
            self._ndim = item.ndim
            self._device = item.device

            self._batch_size = batch_size
            self._padded_shape = padded_shape
            self._padding_value = padding_value

            if self._padded_shape is not None:
                self._padded_shape = [(None if s is not None and s < 1 else s) for s in self._padded_shape]
            else:
                self._padded_shape = [None] * self._ndim

            if self._padding_value is None:
                self._padding_value = 0

        def _batch_insert(self, batch, idx, item):
            if item is None:
                batch[idx] = 0
            else:
                max_shape = batch.shape[1:]

                inner_slices = (idx,) + tuple(slice(0, s, 1) for s in item.shape)
                outer_slices = (idx,) + tuple(slice(s, m, 1) for s, m in zip(item.shape, max_shape))
                all_slices = (idx,) + tuple(slice(0, m, 1) for m in max_shape)

                batch[inner_slices] = item
                for i in range(1, len(outer_slices)):
                    slices = all_slices[:i] + outer_slices[i:i + 1]
                    if i < len(outer_slices) - 1:
                        slices = slices + all_slices[i + 1:]

                    batch[slices] = self._padding_value

        def make_batch(self, items, idx):
            if self._ndim == 0:
                if self._padded_shape is not None:
                    batch = torch.empty([self._batch_size] + self._padded_shape, dtype=self._dtype, device=self._device)

                    for i, item in enumerate(items):
                        item = item[idx]
                        item = torch.full([1] * len(self._padded_shape), item, dtype=self._dtype, device=self._device)
                        self._batch_insert(batch, i, item)
                else:
                    batch = torch.empty([self._batch_size], dtype=self._dtype, device=self._device)
                    for i, item in enumerate(items):
                        item = item[idx]
                        if item is None:
                            batch[i] = 0
                        else:
                            batch[i] = item[idx]
            else:
                shapes = [tuple(s[idx].shape) for s in items]
                max_shape = [(max(shapes, key=operator.itemgetter(i))[i]
                              if self._padded_shape[i] is None
                              else self._padded_shape[i])
                             for i in range(self._ndim)]

                expand_dims = 0
                if len(max_shape) < len(self._padded_shape):
                    expand_dims = len(self._padded_shape) - len(max_shape)
                    max_shape = max_shape + [(self._padded_shape[i] if self._padded_shape[i] is not None else 1)
                                             for i in range(len(max_shape), len(self._padded_shape))]

                batch = torch.empty([self._batch_size] + max_shape, dtype=self._dtype, device=self._device)

                for i, item in enumerate(items):
                    item = item[idx]
                    if expand_dims:
                        item = item.unsqueeze(-1)
                        for _ in range(1, expand_dims):
                            item = item.unsqueeze_(-1)

                    self._batch_insert(batch, i, item)

            if len(items) < self._batch_size:
                batch[len(items):] = 0

            return batch

    _STRATEGIES.insert(0, _TorchStrategy)
except (ImportError, ModuleNotFoundError):
    pass


class _BatchPaddedHelper:
    def __init__(self, batch_size, initial_items, padded_shapes, padding_values):
        super().__init__()

        def chooser(t):
            for s in _STRATEGIES:
                if s.is_supported(t):
                    return s
            else:
                raise ValueError('Unsupported')

        sample = initial_items[0]
        self._width = len(sample)
        self._squeeze = self._width == 1 and not sample.is_tuple

        if padding_values is None:
            padding_values = [None] * self._width

        if padded_shapes is None:
            padded_shapes = [None] * self._width

        self._stategies = [chooser(type(item))(item, batch_size, padded_shapes[i], padding_values[i])
                           for i, item in enumerate(sample.value)]

        for i in range(1, len(initial_items)):
            sample = initial_items[i]
            assert self._width == len(sample)

            is_valid = all([s.is_supported(type(item)) for s, item in zip(self._stategies, sample.value)])

            assert is_valid, f'Sample #{i} is not supported by chosen strategies'

    def make_batch(self, items):
        batch = tuple(s.make_batch(items, i) for i, s in enumerate(self._stategies))
        return batch[0] if self._squeeze else batch


class _SampleWrapper:
    _none = object()

    def __init__(self, sample):
        if sample is self._none:
            self._is_tuple = False
            self._sample = None
            self._is_disposed = True
        else:
            self._is_tuple = isinstance(sample, tuple)
            if not self._is_tuple:
                sample = (sample,)
            self._sample = sample
            self._is_disposed = False

    def __len__(self):
        return len(self._sample)

    def __getitem__(self, idx):
        return self._sample[idx]

    @property
    def is_disposed(self):
        return self._is_disposed

    @property
    def value(self):
        return self._sample

    @property
    def is_tuple(self):
        return self._is_tuple

    @staticmethod
    def next(iter):
        return _SampleWrapper(next(iter, _SampleWrapper._none))


class _BatchPaddedIterator:
    def __init__(self, source_iter, batch_size, padded_shapes, padding_values, drop_last):
        self._source_iter = source_iter
        self._batch_size = batch_size
        self._padded_shapes = padded_shapes
        self._padding_values = padding_values
        self._drop_last = drop_last

        self._source_disposed = False

        self._batch_helper = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._source_disposed:
            batch = []
        else:
            batch = [_SampleWrapper.next(self._source_iter) for _ in range(self._batch_size)]

            self._source_disposed = batch[-1].is_disposed

            if self._source_disposed:
                if self._drop_last:
                    batch.clear()
                else:
                    idx = self._batch_size - 1
                    while idx >= 0:
                        if not batch[idx].is_disposed:
                            break
                        else:
                            idx -= 1
                    batch = batch[:idx + 1]

        if not batch:
            raise StopIteration()
        else:
            if self._batch_helper is None:
                self._batch_helper = _BatchPaddedHelper(
                    self._batch_size, batch, self._padded_shapes, self._padding_values)

            return self._batch_helper.make_batch(batch)


class BatchPaddedDataOperation:
    def __init__(self, *, source, batch_size, padded_shapes, padding_values, drop_last):
        self._source = source
        self._batch_size = batch_size
        self._padded_shapes = padded_shapes
        self._padding_values = padding_values
        self._drop_last = drop_last

    def __iter__(self):
        return _BatchPaddedIterator(
            iter(self._source), self._batch_size, self._padded_shapes, self._padding_values, self._drop_last)
