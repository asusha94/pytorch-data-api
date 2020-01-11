
class _TensorsIterator:
    def __init__(self, tensors):
        self._tensors = tensors

    def __iter__(self):
        return self

    def __next__(self):
        if self._tensors is None:
            raise StopIteration()
        else:
            try:
                return self._tensors
            finally:
                self._tensors = None


class TensorsDataSource:
    def __init__(self, *tensor_args, tensors=None):
        assert bool(len(tensor_args)) != (tensors is not None), \
            'tensors: only one way of initialization is supported'

        if len(tensor_args):
            tensors = tensor_args
        else:
            assert isinstance(tensors, tuple), 'tensors: must be a tuple of tensors'

        self._tensors = tensors

    def __iter__(self):
        return _TensorsIterator(self._tensors)
