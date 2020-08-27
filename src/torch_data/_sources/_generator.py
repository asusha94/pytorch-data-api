import aioitertools


class _GeneratorIterator:
    def __init__(self, session_id, iterator):
        self._session_id = session_id
        self._iter = iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iter is None:
            raise StopAsyncIteration()

        try:
            return await aioitertools.next(self._iter)
        except StopAsyncIteration:
            self._iter = None
            raise


class GeneratorDataSource:
    def __init__(self, *, generator, args=None):
        if args is None:
            args = tuple()

        self._generator = generator
        self._args = args

    def get_iter(self, session_id):
        return _GeneratorIterator(session_id, aioitertools.iter(self._generator(*self._args)))
