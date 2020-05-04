

class _CollateIterator:
    _none = object()

    def __init__(self, source_iter, collate_func, buffer_size):
        self._source_iter = source_iter
        self._collate_func = collate_func
        self._buffer_size = buffer_size

        self._collate_iter = None
        self._iter_disposed = False

    def __iter__(self):
        return self

    def __next__(self):
        while self._source_iter is not None:
            if self._collate_iter is None:
                buffer = []
                while self._source_iter is not None and (self._buffer_size is None or len(buffer) < self._buffer_size):
                    sample = next(self._source_iter, self._none)
                    if sample is self._none:
                        self._source_iter = None
                    else:
                        buffer.append(sample)

                if buffer:
                    self._collate_iter = iter(self._collate_func(buffer))

            if self._collate_iter is None:
                raise StopIteration()
            else:
                sample = next(self._collate_iter, self._none)
                if sample is self._none:
                    self._collate_iter = None
                else:
                    return sample
        else:
            raise StopIteration()


class CollateDataOperation:
    def __init__(self, *, source, collate_func, buffer_size):
        self._source = source
        self._collate_func = collate_func
        self._buffer_size = buffer_size

    def get_iter(self, session_id):
        return _CollateIterator(self._source.get_iter(session_id), self._collate_func, self._buffer_size)
