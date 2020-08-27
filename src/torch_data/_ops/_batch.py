import aioitertools
import copy

_STRATEGIES = []


class _DefaultStrategy:
    @staticmethod
    def is_supported(item_type):
        return True

    def __init__(self, item, batch_size):
        self._batch_size = batch_size

    def make_batch(self):
        return [None] * self._batch_size

    def batch_insert(self, batch, idx, item):
        if item is None:
            batch[idx] = None
        else:
            batch[idx] = copy.deepcopy(item)


_STRATEGIES.append(_DefaultStrategy)

try:
    import numpy as np

    class _NumpyStrategy:
        @staticmethod
        def is_supported(item_type):
            return item_type.__module__ == np.__name__ or item_type in [float, int]

        def __init__(self, item, batch_size):
            if isinstance(item, np.ndarray):
                self._dtype = item.dtype
            else:
                self._dtype = np.dtype(type(item))

            self._shape = [batch_size] + list(np.shape(item))

        def make_batch(self):
            return np.empty(self._shape, dtype=self._dtype)

        def batch_insert(self, batch, idx, item):
            if item is None:
                batch[idx, ...] = 0
            else:
                batch[idx, ...] = item

    _STRATEGIES.insert(0, _NumpyStrategy)
except (ImportError, ModuleNotFoundError):
    pass

try:
    import torch

    class _TorchStrategy:
        @staticmethod
        def is_supported(item_type):
            return item_type == torch.Tensor

        def __init__(self, item, batch_size):
            self._dtype = item.dtype
            self._shape = [batch_size] + list(item.size())
            self._device = item.device

        def make_batch(self):
            return torch.empty(*self._shape, dtype=self._dtype, device=self._device)

        def batch_insert(self, batch, idx, item):
            if item is None:
                batch[idx, ...] = 0
            else:
                batch[idx, ...] = item

    _STRATEGIES.insert(0, _TorchStrategy)
except (ImportError, ModuleNotFoundError):
    pass


class _BatchHelper:
    def __init__(self, batch_size, sample):
        super().__init__()

        def chooser(t):
            for s in _STRATEGIES:
                if s.is_supported(t):
                    return s
            else:
                raise ValueError('Unsupported')

        self._stategies = [chooser(type(item))(item, batch_size) for item in sample]

    def make_batch(self):
        return tuple(s.make_batch() for s in self._stategies)

    def batch_insert(self, batch, idx, sample):
        assert len(sample) == len(self._stategies), ''

        _ = [s.batch_insert(b, idx, i) for b, s, i in zip(batch, self._stategies, sample)]


class _BatchIterator:
    _none = object()

    def __init__(self, source_iter, batch_size, drop_last):
        self._source_iter = source_iter
        self._batch_size = batch_size
        self._drop_last = drop_last

        self._batch = None
        self._batch_counter = 0

        self._batch_helper = None
        self._squeeze = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            while self._batch_counter < self._batch_size:
                try:
                    sample = await aioitertools.next(self._source_iter)

                    is_tuple = isinstance(sample, tuple)
                    if not is_tuple:
                        sample = (sample,)

                    if self._batch_helper is None:
                        self._batch_helper = _BatchHelper(self._batch_size, sample)
                        self._squeeze = not is_tuple

                    if self._batch is None:
                        self._batch = self._batch_helper.make_batch()

                    assert len(self._batch) == len(sample), ''

                    self._batch_helper.batch_insert(self._batch, self._batch_counter, sample)

                    self._batch_counter += 1
                except StopAsyncIteration:
                    if self._drop_last:
                        self._batch = None
                    elif self._batch is not None:
                        sample = tuple([None] * len(self._batch))
                        for i in range(self._batch_counter, self._batch_size):
                            self._batch_helper.batch_insert(self._batch, i, sample)

                    self._batch_counter = self._batch_size

            if self._batch is None:
                raise StopAsyncIteration()
            else:
                return self._batch[0] if self._squeeze else self._batch
        finally:
            self._batch_counter = 0
            self._batch = None


class BatchDataOperation:
    def __init__(self, *, source, batch_size, drop_last):
        self._source = source
        self._batch_size = batch_size
        self._drop_last = drop_last

    def get_iter(self, session_id):
        return _BatchIterator(self._source.get_iter(session_id), self._batch_size, self._drop_last)
