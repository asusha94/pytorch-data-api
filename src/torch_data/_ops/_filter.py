import asyncio
import aioitertools


class _Iterator:
    def __init__(self, source_iter, predicate, expand_args):
        self._source_iter = source_iter

        if asyncio.iscoroutinefunction(predicate):
            self._predicate = predicate
        else:
            async def _wrapper(*args):
                return predicate(*args)

            self._predicate = _wrapper

        self._expand_args = expand_args

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._source_iter is not None:
            try:
                sample = await aioitertools.next(self._source_iter)
                if not self._expand_args:
                    val = await self._predicate(sample)
                else:
                    if not isinstance(sample, tuple):
                        sample = (sample,)
                    val = await self._predicate(*sample)

                if val:
                    return sample
            except StopAsyncIteration:
                self._source_iter = None
                raise
        else:
            raise StopAsyncIteration()


class FilterDataOperation:
    def __init__(self, *, source, predicate, expand_args):
        self._source = source
        self._predicate = predicate
        self._expand_args = expand_args

    def get_iter(self, session_id):
        return _Iterator(self._source.get_iter(session_id), self._predicate, self._expand_args)
