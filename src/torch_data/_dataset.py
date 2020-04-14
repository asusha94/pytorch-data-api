
from ._sources import GeneratorDataSource, TensorSlicesDataSource, TensorsDataSource
from ._sources import ConcatenateDataSource, InterleaveDataSource

from ._ops import (BatchDataOperation, BatchPaddedDataOperation, CollateDataOperation,
                   FilterDataOperation, MapDataOperation, ShuffleDataOperation,
                   UnBatchDataOperation, WindowDataOperation, WindowPaddedDataOperation,
                   PrefetchDataOperation)


class _EmptyDatasetIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration()


class _DatasetIterator:
    _none = object()

    def __init__(self, dataset_iter):
        self._dataset_iter = dataset_iter

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._dataset_iter, self._none)
        if item is self._none:
            raise StopIteration()
        else:
            return item


class Dataset:
    @staticmethod
    def from_generator(generator, args=None):
        assert callable(generator), 'generator: Must be callable'
        assert args is None or isinstance(args, (list, tuple)), 'args: Must be None or a tuple'

        source = GeneratorDataSource(generator=generator, args=args)
        return Dataset(_impl=source)

    @staticmethod
    def from_tensor_slices(*tensor_args, tensors=None):
        source = TensorSlicesDataSource(*tensor_args, tensors=tensors)
        return Dataset(_impl=source)

    @staticmethod
    def from_tensors(*tensor_args, tensors=None):
        source = TensorsDataSource(*tensor_args, tensors=tensors)
        return Dataset(_impl=source)

    @staticmethod
    def concatenate(*dataset_args, datasets=None):
        if datasets is None:
            datasets = dataset_args

        assert isinstance(datasets, (list, tuple)) and len(datasets), \
            'datasets: must be a non-empty instance of a list or tuple'
        assert all([isinstance(d, Dataset) for d in datasets]), \
            'datasets: all arguments must be an instance of Dataset class'

        if len(datasets) == 1:
            return datasets[0]
        else:
            source = ConcatenateDataSource(*datasets)
            return Dataset(_impl=source)

    @staticmethod
    def interleave(*dataset_args, datasets=None):
        if datasets is None:
            datasets = dataset_args

        assert isinstance(datasets, (list, tuple)) and len(datasets), \
            'datasets: must be a non-empty instance of a list or tuple'
        assert all([isinstance(d, Dataset) for d in datasets]), \
            'datasets: all arguments must be an instance of Dataset class'

        if len(datasets) == 1:
            return datasets[0]
        else:
            source = InterleaveDataSource(*datasets)
            return Dataset(_impl=source)

    #
    # operations

    def batch(self, batch_size, *, drop_last=True):
        assert isinstance(batch_size, int), 'batch_size: must be an integer'
        assert isinstance(drop_last, bool), 'drop_last: must be a boolean'

        op = BatchDataOperation(source=self._impl, batch_size=batch_size, drop_last=drop_last)
        return Dataset(_impl=op)

    def batch_padded(self, batch_size, *, padded_shapes=None, padding_values=None, drop_last=True):
        assert isinstance(batch_size, int), 'batch_size: must be an integer'
        assert isinstance(drop_last, bool), 'drop_last: must be a boolean'

        op = BatchPaddedDataOperation(source=self._impl, batch_size=batch_size,
                                      padded_shapes=padded_shapes,
                                      padding_values=padding_values, drop_last=drop_last)
        return Dataset(_impl=op)

    def collate(self, collate_func, buffer_size=None):
        assert callable(collate_func), 'collate_func: Must be callable'
        assert buffer_size is None or isinstance(buffer_size, int), 'buffer_size: must be an integer'
        assert buffer_size is None or buffer_size > 2, 'buffer_size: must be greater than 2'

        op = CollateDataOperation(source=self._impl, collate_func=collate_func, buffer_size=buffer_size)
        return Dataset(_impl=op)

    def filter(self, predicate, expand_args=True):
        assert callable(predicate), 'predicate: Must be callable'

        op = FilterDataOperation(source=self._impl, predicate=predicate, expand_args=expand_args)

        return Dataset(_impl=op)

    def map(self, map_func, num_parallel_calls=None, ordered=True, ignore_errors=False):
        assert callable(map_func), 'map_func: Must be callable'
        assert num_parallel_calls is None or isinstance(num_parallel_calls, int), \
            'num_parallel_calls: Must be None or integer'

        op = MapDataOperation(source=self._impl, map_func=map_func,
                              num_parallel_calls=num_parallel_calls,
                              ordered=ordered, ignore_errors=ignore_errors)

        return Dataset(_impl=op)

    def shuffle(self, buffer_size, seed=None):
        assert isinstance(buffer_size, int), 'buffer_size: must be an integer'
        assert buffer_size > 1, 'buffer_size: must be greater than 1'

        op = ShuffleDataOperation(source=self._impl, buffer_size=buffer_size, seed=seed)
        return Dataset(_impl=op)

    def unbatch(self):
        op = UnBatchDataOperation(source=self._impl)
        return Dataset(_impl=op)

    def window(self, size, stride=1, *, drop_last=True):
        assert isinstance(size, int), 'size: must be an integer'
        assert isinstance(stride, int), 'stride: must be an integer'
        assert isinstance(drop_last, bool), 'drop_last: must be a boolean'

        op = WindowDataOperation(source=self._impl, size=size, stride=stride, drop_last=drop_last)
        return Dataset(_impl=op)

    def window_padded(self, size, stride=1, *, padded_shapes=None, padding_values=None, drop_last=True):
        assert isinstance(size, int), 'size: must be an integer'
        assert isinstance(stride, int), 'stride: must be an integer'
        assert isinstance(drop_last, bool), 'drop_last: must be a boolean'

        op = WindowPaddedDataOperation(source=self._impl, size=size, stride=stride,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values, drop_last=drop_last)
        return Dataset(_impl=op)

    def prefetch(self, size):
        assert isinstance(size, int), 'size: must be an integer'

        op = PrefetchDataOperation(source=self._impl, buffer_size=size)
        return Dataset(_impl=op)

    # def repeat(self, times=None):
    #     pass

    #
    #
    #

    def __init__(self, *, _impl=None):
        self.__impl = _impl
        if self.__impl is None:
            self.__get_iterator = lambda: _EmptyDatasetIterator()
        else:
            self.__get_iterator = lambda: _DatasetIterator(iter(self._impl))

    @property
    def _impl(self):
        return self.__impl

    def __iter__(self):
        return self.__get_iterator()
