import asyncio
import aioitertools
import collections
import dill
import multiprocessing as mp


class _SerialIterator:
    def __init__(self, source, map_func, *, ignore_errors=False):
        self._source_iter = source

        if asyncio.iscoroutinefunction(map_func):
            self._map_func = map_func
        else:
            async def _wrapper(*args, **kwargs):
                return map_func(*args, **kwargs)

            self._map_func = _wrapper

        self._ignore_errors = ignore_errors

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._source_iter is None:
            raise StopAsyncIteration()
        else:
            try:
                sample = await aioitertools.next(self._source_iter)
                if not isinstance(sample, tuple):
                    sample = (sample,)

                try:
                    return await self._map_func(*sample)
                except Exception:
                    if not self._ignore_errors:
                        raise
                    else:
                        import sys
                        import traceback
                        traceback.print_exc(file=sys.stderr)

                        return None
            except StopAsyncIteration:
                self._source_iter = None
                raise


class AsyncProcess:
    _Task = collections.namedtuple('Task', ['coro', 'args', 'kwargs'])

    @staticmethod
    def _process_main(queue, cancel_token):
        loop = asyncio.get_event_loop()

        async def token_checker():
            while not cancel_token.is_set():
                await asyncio.sleep(0.05)

        async def queue_checker():
            while True:
                if queue.empty():
                    await asyncio.sleep(0)
                else:
                    task = dill.loads(queue.get())
                    fut = loop.create_task(task.coro(*task.args, **task.kwargs))
                    fut.add_done_callback(lambda f: queue.task_done())

        def token_callback(f):
            if not loop.is_closed():
                if loop.is_running():
                    for task in asyncio.all_tasks():
                        task.cancel()
                loop.stop()

        loop.create_task(token_checker()).add_done_callback(token_callback)
        loop.create_task(queue_checker())

        try:
            loop.run_forever()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            if not loop.is_closed():
                loop.close()

    @property
    def n_tasks(self):
        return self._tasks_queue._unfinished_tasks.get_value()

    def __init__(self, mp_ctx):
        self._tasks_queue = mp_ctx.JoinableQueue()
        self._tasks_queue.cancel_join_thread()
        self._cancel_token = mp_ctx.Event()

        self._process = mp_ctx.Process(target=AsyncProcess._process_main,
                                       args=(self._tasks_queue, self._cancel_token))
        self._process.daemon = True

    def __del__(self):
        self.stop()

    def start(self):
        self._process.start()

    def stop(self):
        self._cancel_token.set()

    def join(self):
        if self._process.is_alive():
            self._process.join()

    def add_task(self, task, args=None, kwargs=None):
        assert self._process.is_alive()
        assert asyncio.iscoroutinefunction(task)

        if args is None:
            args = tuple()

        if kwargs is None:
            kwargs = dict()

        task = AsyncProcess._Task(coro=task, args=args, kwargs=kwargs)

        self._tasks_queue.put(dill.dumps(task))


class _ParallelSession:
    class Pool:
        @property
        def uid(self):
            return self._uid

        def __init__(self, uid, ctx):
            self.ctx = ctx

            self._uid = uid
            self._counter = 0

            self._pool = None

        def submit(self, task, args=None, kwargs=None):
            assert self._pool is not None, 'The Pool has been disposed'

            if args is None:
                args = tuple()

            if kwargs is None:
                kwargs = dict()

            process = min(self._pool, key=lambda x: x.n_tasks)
            return process.add_task(task, args=args, kwargs=kwargs)

        def _start(self):
            if not self._pool:
                import os
                self._pool = [AsyncProcess(self.ctx) for _ in range(os.cpu_count())]

                for t in self._pool:
                    t.start()

        def _stop(self):
            if self._pool:
                for t in self._pool:
                    t.stop()

                for t in self._pool:
                    t.join()

                self._pool.clear()
                self._pool = None

        def _increment_ref(self):
            self._counter += 1
            if self._counter > 0:
                self._start()

        def _decrement_ref(self):
            self._counter -= 1
            assert self._counter >= 0

            if self._counter == 0:
                self._stop()

            return self._counter == 0

    _pools = dict()

    manager = mp.Manager()

    @staticmethod
    def get(uid):
        pool = _ParallelSession._pools.get(uid)
        if pool is None:
            pool = _ParallelSession.Pool(uid, mp.get_context('spawn'))
            _ParallelSession._pools[uid] = pool

        pool._increment_ref()

        return pool

    @staticmethod
    def release(pool):
        if pool._decrement_ref():
            del _ParallelSession._pools[pool.uid]

# TODO : implement PARALELL


class _ParallelIterator:
    _none = object()

    @staticmethod
    async def _parallel_process(
            map_func, input_queue, input_queue_n_tasks,
            output_queue, output_queue_n_tasks, cancel_token, ignore_errors=False):
        import multiprocessing

        _map_func = dill.loads(map_func)

        if asyncio.iscoroutinefunction(map_func):
            map_func = _map_func
        else:
            async def _wrapper(*args, **kwargs):
                return _map_func(*args, **kwargs)

            map_func = _wrapper

        while not cancel_token.is_set():
            if input_queue.empty():
                await asyncio.sleep(0)
            else:
                try:
                    sample = dill.loads(input_queue.get())

                    try:
                        result = await map_func(*sample)
                        while output_queue.full():
                            await asyncio.sleep(0)
                        else:
                            output_queue.put(dill.dumps(result))
                            output_queue_n_tasks._callmethod('release')
                    except Exception:
                        import sys
                        import traceback
                        print(multiprocessing.current_process().name, 'got an error:\n', traceback.format_exc(),
                              file=sys.stderr, flush=True)

                        if not ignore_errors:
                            raise
                finally:
                    input_queue_n_tasks._callmethod('acquire')

    def __init__(self, session_id, source, map_func, n_workers, ordered, *, ignore_errors=False):
        self._pool = _ParallelSession.get(session_id)

        self._stop_fetching = False

        self._n_workers = n_workers
        self._source_iter = source

        map_func_dump = dill.dumps(map_func)

        self._input_queue = _ParallelSession.manager.Queue()
        self._input_queue_n_tasks = _ParallelSession.manager.Semaphore(0)
        self._output_queue = _ParallelSession.manager.Queue()
        self._output_queue_n_tasks = _ParallelSession.manager.Semaphore(0)
        self._cancel_token = _ParallelSession.manager.Event()

        self._ignore_errors = ignore_errors

        self._queue = []

        self._ordered = ordered

        for _ in range(n_workers):
            self._pool.submit(
                _ParallelIterator._parallel_process,
                args=(map_func_dump, self._input_queue, self._input_queue_n_tasks,
                      self._output_queue, self._output_queue_n_tasks, self._cancel_token, ignore_errors)
            )

    def __del__(self):
        if self._pool is not None:
            self._cancel_token.set()
            _ParallelSession.release(self._pool)
            self._pool = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._source_iter is not None or (self._input_queue_n_tasks._callmethod('get_value') > 0 or
                                                self._output_queue_n_tasks._callmethod('get_value') > 0):
            if not self._input_queue.full():
                try:
                    sample = await aioitertools.next(self._source_iter)
                    if not isinstance(sample, tuple):
                        sample = (sample,)
                    self._input_queue.put(dill.dumps(sample))
                    self._input_queue_n_tasks._callmethod('release')

                except StopAsyncIteration:
                    self._source_iter = None

                if not self._output_queue.empty():
                    # TODO : ordering
                    try:
                        result = dill.loads(self._output_queue.get())
                        return result
                    finally:
                        self._output_queue_n_tasks._callmethod('acquire')

            await asyncio.sleep(0)


class MapDataOperation:
    def __init__(self, *, source, map_func, num_parallel_calls, ordered=True, ignore_errors=False):
        num_parallel_calls = 0
        if num_parallel_calls == 0:
            self._get_iterator = lambda sid: _SerialIterator(
                source.get_iter(sid), map_func, ignore_errors=ignore_errors)
        else:
            self._get_iterator = lambda sid: _ParallelIterator(sid,
                                                               source.get_iter(sid), map_func,
                                                               n_workers=num_parallel_calls,
                                                               ordered=ordered,
                                                               ignore_errors=ignore_errors)

    def get_iter(self, session_id):
        return self._get_iterator(session_id)
