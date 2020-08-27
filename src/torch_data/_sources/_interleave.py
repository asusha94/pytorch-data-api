import aioitertools


class _InterleaveIterator:
    def __init__(self, session_id, dataset_iters, drop_tails):
        self._session_id = session_id
        self._dataset_iters = dataset_iters

        self._drop_tails = drop_tails
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        while len(self._dataset_iters):
            try:
                return await aioitertools.next(self._dataset_iters[self._idx])
            except StopAsyncIteration:
                if self._drop_tails:
                    self._dataset_iters.clear()
                else:
                    del self._dataset_iters[self._idx]
                    self._idx -= 1
            finally:  # cyclic
                self._idx += 1
                if self._idx >= len(self._dataset_iters):
                    self._idx = 0
        else:
            raise StopAsyncIteration()


class InterleaveDataSource:
    def __init__(self, *, dataset_sources, drop_tails):
        self._dataset_sources = dataset_sources
        self._drop_tails = drop_tails

    def get_iter(self, session_id):
        sources = [d.get_iter(session_id) for d in self._dataset_sources]
        return _InterleaveIterator(session_id, sources, self._drop_tails)
