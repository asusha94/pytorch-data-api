import asyncio
import aioitertools


class _CollateIterator:
    def __init__(self, source_iter, collate_func, buffer_size):
        self._source_iter = source_iter

        if asyncio.iscoroutinefunction(collate_func):
            self._collate_func = collate_func
        else:
            async def _wrapper(*args):
                return collate_func(*args)

            self._collate_func = _wrapper

        self._buffer_size = buffer_size

        self._collate_iter = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._source_iter is not None:
            if self._collate_iter is None:
                buffer = []
                while self._source_iter is not None and (self._buffer_size is None or len(buffer) < self._buffer_size):
                    try:
                        sample = await aioitertools.next(self._source_iter)
                        buffer.append(sample)
                    except StopAsyncIteration:
                        self._source_iter = None

                if buffer:
                    self._collate_iter = aioitertools.iter(await self._collate_func(buffer))

            if self._collate_iter is None:
                raise StopAsyncIteration()
            else:
                try:
                    return await aioitertools.next(self._collate_iter)
                except StopAsyncIteration:
                    self._collate_iter = None
        else:
            raise StopAsyncIteration()


class CollateDataOperation:
    def __init__(self, *, source, collate_func, buffer_size):
        self._source = source
        self._collate_func = collate_func
        self._buffer_size = buffer_size

    def get_iter(self, session_id):
        return _CollateIterator(self._source.get_iter(session_id), self._collate_func, self._buffer_size)
