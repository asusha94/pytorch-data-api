
from ._batch import _BatchHelper


class _WindowIterator:
    _none = object()

    def __init__(self, source_iter, size, stride, drop_last):
        self._source_iter = source_iter
        self._size = size
        self._stride = stride
        self._drop_last = drop_last

        self._skip = 0

        self._window = []

        self._batch_helper = None
        self._squeeze = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while len(self._window) < self._size:
                while self._skip > 0:
                    self._skip -= 1
                    sample = next(self._source_iter, self._none)
                    if sample is self._none:
                        break
                else:
                    sample = next(self._source_iter, self._none)

                if sample is self._none:
                    if self._drop_last:
                        self._window.clear()
                    break
                else:
                    is_tuple = isinstance(sample, tuple)
                    if not is_tuple:
                        sample = (sample,)

                    if self._batch_helper is None:
                        self._batch_helper = _BatchHelper(self._size, sample)
                        self._squeeze = not is_tuple

                    self._window.append(sample)

            if not len(self._window):
                raise StopIteration()
            else:
                batch = self._batch_helper.make_batch()
                _ = [self._batch_helper.batch_insert(batch, i, sample) for i, sample in enumerate(self._window)]
                return batch[0] if self._squeeze else batch
        finally:
            if self._stride < len(self._window):
                del self._window[:self._stride]
            else:
                self._skip = self._stride - len(self._window)
                self._window.clear()


class WindowDataOperation:
    def __init__(self, *, source, size, stride, drop_last):
        self._source = source
        self._size = size
        self._stride = stride
        self._drop_last = drop_last

    def __iter__(self):
        return _WindowIterator(iter(self._source), self._size, self._stride, self._drop_last)
