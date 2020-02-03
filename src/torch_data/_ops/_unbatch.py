
class _UnBatchIterator:
    _none = object()

    def __init__(self, source_iter):
        self._source_iter = source_iter
        self._batch_size = batch_size
        self._drop_last = drop_last

        self._batch = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._batch is None:
                sample = next(self._source_iter, self._none)
                if sample is self._none:
                    raise StopIteration()
                else:
                    self._batch = sample

            sample = next(self._batch, self._none)
            if sample is self._none:
                self._batch = None
            else:
                return sample


class UnBatchDataOperation:
    def __init__(self, *, source):
        self._source = source

    def __iter__(self):
        return _UnBatchIterator(iter(self._source))
