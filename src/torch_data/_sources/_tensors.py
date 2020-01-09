
class TensorsIterator:
    def __init__(self, tensors):
        self._tensors = tensors
        self._i = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._i += 1
        if self._i > 0:
            raise StopIteration()
        else:
            return self._tensors


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
        return TensorsIterator(self._tensors)
