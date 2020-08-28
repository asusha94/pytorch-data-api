import asyncio
import aioitertools
import threading
import queue


# class AsyncThread:
#     @staticmethod
#     def _thread_main(loop, cancel_token):
#         async def token_checker():
#             while not cancel_token.is_set():
#                 await asyncio.sleep(0.05)

#         def token_callback(f):
#             if not loop.is_closed():
#                 if loop.is_running():
#                     for task in asyncio.all_tasks():
#                         task.cancel()
#                 loop.stop()

#         # asyncio.set_event_loop(loop)

#         loop.create_task(token_checker()).add_done_callback(token_callback)

#         try:
#             loop.run_forever()
#         except Exception:
#             import traceback
#             traceback.print_exc()
#         finally:
#             if not loop.is_closed():
#                 loop.close()

#     @property
#     def n_tasks(self):
#         return self._n_tasks

#     def __init__(self):
#         self._loop = asyncio.new_event_loop()
#         self._cancel_token = threading.Event()

#         self._thread = threading.Thread(target=AsyncThread._thread_main, args=(self._loop, self._cancel_token))
#         self._thread.daemon = True
#         self._n_tasks = 0

#     def __del__(self):
#         self.stop()

#     def start(self):
#         self._thread.start()

#     def stop(self):
#         self._cancel_token.set()

#     def join(self):
#         if self._thread.is_alive():
#             self._thread.join()

#     def add_task(self, task, args=None, kwargs=None):
#         assert self._thread.is_alive()
#         assert asyncio.iscoroutinefunction(task)

#         if args is None:
#             args = tuple()

#         if kwargs is None:
#             kwargs = dict()

#         fut = asyncio.run_coroutine_threadsafe(task(*args, **kwargs), self._loop)
#         fut.add_done_callback(lambda f: self._dec_n_tasks())

#         self._n_tasks += 1
#         return fut

#     def _dec_n_tasks(self):
#         self._n_tasks -= 1


# class _ParallelSession:
#     class Pool:
#         @property
#         def uid(self):
#             return self._uid

#         def __init__(self, uid):
#             self._uid = uid
#             self._counter = 0

#             self._pool = None

#         def submit(self, task, args=None, kwargs=None):
#             assert self._pool is not None, 'The Pool has been disposed'

#             thread = min(self._pool, key=lambda x: x.n_tasks)
#             return thread.add_task(task, args=args, kwargs=kwargs)

#         def _start(self):
#             if not self._pool:
#                 import os
#                 # self._pool = [AsyncThread() for _ in range(os.cpu_count())]
#                 self._pool = [AsyncThread() for _ in range(1)]

#                 for t in self._pool:
#                     t.start()

#         def _stop(self):
#             if self._pool:
#                 for t in self._pool:
#                     t.stop()

#                 # for t in self._pool:
#                 #     t.join()

#                 self._pool.clear()
#                 self._pool = None

#         def _increment_ref(self):
#             self._counter += 1
#             if self._counter > 0:
#                 self._start()

#         def _decrement_ref(self):
#             self._counter -= 1
#             assert self._counter >= 0

#             if self._counter == 0:
#                 self._stop()

#             return self._counter == 0

#     _pools = dict()

#     @staticmethod
#     def get(uid):
#         pool = _ParallelSession._pools.get(uid)
#         if pool is None:
#             pool = _ParallelSession.Pool(uid)
#             _ParallelSession._pools[uid] = pool

#         pool._increment_ref()

#         return pool

#     @staticmethod
#     def release(pool):
#         if pool._decrement_ref():
#             del _ParallelSession._pools[pool.uid]


class _PrefetchIterator:
    _none = object()

    @staticmethod
    async def _prefetch_fn(output_queue, source_iter, cancel_token):
        while not cancel_token.is_set():
            try:
                sample = await aioitertools.next(source_iter)
            except StopAsyncIteration:
                sample = _PrefetchIterator._none
                cancel_token.set()

            while output_queue.full():
                await asyncio.sleep(0)
            else:
                output_queue.put(sample)

    def __init__(self, session_id, source_iter, buffer_size):
        self._source_iter = source_iter

        buffer = queue.Queue(buffer_size)
        self._buffer = buffer

        cancel_token = threading.Event()
        self._cancel_token = cancel_token

        self._task = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._task is None:
            self._task = asyncio.get_event_loop().create_task(
                _PrefetchIterator._prefetch_fn(self._buffer, self._source_iter, self._cancel_token))

        if self._buffer is None:
            raise StopAsyncIteration
        else:
            while self._buffer.empty():
                await asyncio.sleep(0)
            else:
                sample = self._buffer.get()
                self._buffer.task_done()

            if sample is self._none:
                self._buffer = None

                if self._cancel_token is not None:
                    self._cancel_token.set()
                    self._cancel_token = None
                    await self._task
                raise StopAsyncIteration
            else:
                return sample


class PrefetchDataOperation:
    def __init__(self, *, source, buffer_size):
        self._source = source
        self._buffer_size = buffer_size

    def get_iter(self, session_id):
        return _PrefetchIterator(session_id, self._source.get_iter(session_id), self._buffer_size)
