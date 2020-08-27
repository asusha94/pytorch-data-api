import aioitertools


class _ConcatenateIterator:
    def __init__(self, session_id, dataset_iters):
        self._session_id = session_id
        self._dataset_iters = dataset_iters

    def __aiter__(self):
        return self

    async def __anext__(self):
        while len(self._dataset_iters):
            try:
                return await aioitertools.next(self._dataset_iters[0])
            except StopAsyncIteration:
                del self._dataset_iters[0]
        else:
            raise StopAsyncIteration()


class ConcatenateDataSource:
    def __init__(self, *, dataset_sources):
        self._dataset_sources = dataset_sources

    def get_iter(self, session_id):
        return _ConcatenateIterator(
            session_id,
            [d.get_iter(session_id) for d in self._dataset_sources]
        )
