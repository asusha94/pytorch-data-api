import copy
from collections.abc import Iterable


class _SingleTensorSlicesIterator:
    _none = object()

    def __init__(self, tensor_iter):
        self._tensor_iter = tensor_iter

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self._tensor_iter, self._none)
        if sample is self._none:
            raise StopIteration()
        else:
            return sample


class _MultiTensorSlicesIterator:
    _none = object()

    def __init__(self, tensor_iters):
        self._tensor_iters = tensor_iters

    def __iter__(self):
        return self

    def __next__(self):
        sample = tuple(next(t, self._none) for t in self._tensor_iters)
        if any(map(lambda s: s is self._none, sample)):
            raise StopIteration()
        else:
            return tuple(copy.deepcopy(s) for s in sample)


class TensorSlicesDataSource:
    def __init__(self, *tensor_args, tensors=None):
        assert bool(len(tensor_args)) != (tensors is not None), \
            'tensors: only one way of initialization is supported'

        if len(tensor_args):
            tensors = tensor_args
        else:
            assert isinstance(tensors, tuple), 'tensors: must be a tuple of tensors'

        states = set([isinstance(t, Iterable) for t in tensors])
        assert len(states) == 1, 'tensors: all tensors in the tuple must have the same length'

        self._tensors = tensors

        if len(self._tensors) == 1:
            self.__get_iterator = lambda: _SingleTensorSlicesIterator(iter(self._tensors[0]))
        else:
            self.__get_iterator = lambda: _MultiTensorSlicesIterator([iter(t) for t in self._tensors])

    def __iter__(self):
        return self.__get_iterator()
