from collections.abc import Iterable


class TensorSlicesIterator:
    _none = object()

    def __init__(self, tensors):
        self._tensors = tensors
        self._tensors_iters = [iter(t) for t in self._tensors]

    def __iter__(self):
        return self

    def __next__(self):
        sample = tuple(next(t, self._none) for t in self._tensors_iters)
        if any(map(lambda s: s is self._none, sample)):
            raise StopIteration()
        else:
            return sample


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

    def __iter__(self):
        return TensorSlicesIterator(self._tensors)
