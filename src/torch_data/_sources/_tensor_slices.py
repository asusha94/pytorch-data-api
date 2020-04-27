import copy


class _SingleTensorSlicesIterator:
    _none = object()

    def __init__(self, session_id, tensor_iter):
        self._session_id = session_id
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

    def __init__(self, session_id, tensor_iters):
        self._session_id = session_id
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
    def __init__(self, *, tensors=None):
        self._tensors = tensors

        if len(self._tensors) == 1:
            self.__get_iterator = lambda sid: _SingleTensorSlicesIterator(sid, iter(self._tensors[0]))
        else:
            self.__get_iterator = lambda sid: _MultiTensorSlicesIterator(sid, [iter(t) for t in self._tensors])

    def get_iter(self, session_id):
        return self.__get_iterator(session_id)
