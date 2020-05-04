
class _PrefetchIterator:
    _none = object()

    def __init__(self, source_iter, buffer_size):
        self._source_iter = source_iter
        self._buffer_size = buffer_size
        self._buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        while self._source_iter is not None and len(self._buffer) < self._buffer_size:
            sample = next(self._source_iter, self._none)
            if sample is self._none:
                self._source_iter = None
            else:
                self._buffer.append(sample)

        if not self._buffer:
            raise StopIteration()
        else:
            return self._buffer.pop(0)


class PrefetchDataOperation:
    def __init__(self, *, source, buffer_size):
        self._source = source
        self._buffer_size = buffer_size

    def get_iter(self, session_id):
        return _PrefetchIterator(self._source.get_iter(session_id), self._buffer_size)
