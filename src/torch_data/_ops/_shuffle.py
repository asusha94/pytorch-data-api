import aioitertools
import random


class _ShuffleIterator:
    def __init__(self, source_iter, buffer_size, rand):
        self._source_iter = source_iter
        self._buffer_size = buffer_size
        self._rand = rand

        self._buffer = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._source_iter is not None and len(self._buffer) < self._buffer_size:
            try:
                sample = await aioitertools.next(self._source_iter)
                cur_len = len(self._buffer)
                in_idx = self._rand.randint(0, cur_len)  # including
                self._buffer.insert(in_idx, sample)
            except StopAsyncIteration:
                self._source_iter = None

        if not self._buffer:
            raise StopAsyncIteration()
        else:
            return self._buffer.pop(0)


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
