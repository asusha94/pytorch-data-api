import random


class _ShuffleIterator:
    _none = object()

    def __init__(self, source_iter, buffer_size, rand):
        self._source_iter = source_iter
        self._buffer_size = buffer_size
        self._rand = rand

        self._buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        while self._source_iter is not None and len(self._buffer) < self._buffer_size:
            sample = next(self._source_iter, self._none)
            if sample is self._none:
                self._source_iter = None
            else:
                cur_len = len(self._buffer)
                in_idx = self._rand.randint(0, cur_len)  # including
                self._buffer.insert(in_idx, sample)

        if not self._buffer:
            raise StopIteration()
        else:
            s = self._buffer.pop(0)
            return s


class ShuffleDataOperation:
    def __init__(self, *, source, buffer_size, seed=None):
        self._source = source
        self._buffer_size = buffer_size

        if seed is None:
            self._rand = random
        else:
            self._rand = random.Random(seed)

    def get_iter(self, session_id):
        return _ShuffleIterator(self._source.get_iter(session_id), self._buffer_size, self._rand)
