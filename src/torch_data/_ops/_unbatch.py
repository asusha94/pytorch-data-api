import copy


class _SampleIterator:
    _none = object()

    def __init__(self, sample):
        self._iters = [iter(t) for t in sample]

    def __iter__(self):
        return self

    def __next__(self):
        sample = tuple(next(t, self._none) for t in self._iters)
        if any(map(lambda s: s is self._none, sample)):
            raise StopIteration()
        else:
            return tuple(copy.deepcopy(s) for s in sample)


class _UnBatchIterator:
    _none = object()

    def __init__(self, source_iter):
        self._source_iter = source_iter

        self._batch = None
        self._squeeze = None

    def __iter__(self):
        return self

    def __next__(self):
        while self._source_iter is not None:
            if self._batch is None:
                sample = next(self._source_iter, self._none)
                if sample is self._none:
                    self._source_iter = None
                    raise StopIteration()
                else:
                    is_tuple = isinstance(sample, tuple)
                    if not is_tuple:
                        sample = (sample,)

                    self._batch = _SampleIterator(sample)
                    self._squeeze = not is_tuple

            sample = next(self._batch, self._none)
            if sample is self._none:
                self._batch = None
            else:
                return sample[0] if self._squeeze else sample
        else:
            raise StopIteration()


class UnBatchDataOperation:
    def __init__(self, *, source):
        self._source = source

    def get_iter(self, session_id):
        return _UnBatchIterator(self._source.get_iter(session_id))
