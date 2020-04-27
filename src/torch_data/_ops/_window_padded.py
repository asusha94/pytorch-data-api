
from ._batch_padded import _BatchPaddedHelper, _SampleWrapper


class _WindowPaddedIterator:
    def __init__(self, source_iter, size, stride, padded_shapes, padding_values, drop_last):
        self._source_iter = source_iter
        self._size = size
        self._stride = stride
        self._padded_shapes = padded_shapes
        self._padding_values = padding_values
        self._drop_last = drop_last

        self._skip = 0

        self._window = []

        self._source_disposed = False

        self._batch_helper = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while not self._source_disposed and len(self._window) < self._size:
                while self._skip > 0:
                    self._skip -= 1
                    sample = _SampleWrapper.next(self._source_iter)
                    if sample.is_disposed:
                        break
                else:
                    sample = _SampleWrapper.next(self._source_iter)

                if sample.is_disposed:
                    self._source_disposed = True
                    if self._drop_last:
                        self._window.clear()
                else:
                    self._window.append(sample)

            if not len(self._window):
                raise StopIteration()
            else:
                if self._batch_helper is None:
                    self._batch_helper = _BatchPaddedHelper(
                        self._size, self._window, self._padded_shapes, self._padding_values)

                return self._batch_helper.make_batch(self._window)
        finally:
            if self._stride < len(self._window):
                del self._window[:self._stride]
            else:
                self._skip = self._stride - len(self._window)
                self._window.clear()


class WindowPaddedDataOperation:
    def __init__(self, *, source, size, stride, padded_shapes, padding_values, drop_last):
        self._source = source
        self._size = size
        self._stride = stride
        self._padded_shapes = padded_shapes
        self._padding_values = padding_values
        self._drop_last = drop_last

    def get_iter(self, session_id):
        return _WindowPaddedIterator(
            self._source.get_iter(session_id),
            self._size, self._stride, self._padded_shapes, self._padding_values, self._drop_last)
