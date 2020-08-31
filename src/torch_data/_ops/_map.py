import asyncio
import aioitertools
import collections
import dill
import sys
from contextlib import suppress
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
        self._result_ds = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        from .. import _dataset

        while self._source_iter is not None or self._result_ds is not None:
            if self._result_ds is not None:
                try:
                    return await aioitertools.next(self._result_ds)
                except StopAsyncIteration:
                    self._result_ds = None
            else:
                try:
                    sample = await aioitertools.next(self._source_iter)
                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    try:
                        result = await self._map_func(*sample)
                        if isinstance(result, _dataset.Dataset):
                            self._result_ds = aioitertools.iter(result)
                        else:
                            return result
                    except Exception:
                        if not self._ignore_errors:
                            raise
                        else:
                            import sys
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                except StopAsyncIteration:
                    self._source_iter = None
        else:
            raise StopAsyncIteration()


_MP_CTX = mp.get_context('spawn')


class AsyncProcess:
    _Task = collections.namedtuple('Task', ['coro', 'args', 'kwargs'])

    class ProcessImpl(_MP_CTX.Process):
        def __init__(self, queue, cancel_token):
            super().__init__(daemon=True)

            self._queue = queue
            self._cancel_token = cancel_token

        def run(self):
            from signal import SIGINT, SIGTERM

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _wrapper(coro, *args, **kwargs):
                try:
                    await coro(*args, **kwargs)
                except (EOFError, BrokenPipeError, ConnectionResetError, FileNotFoundError):
                    pass

            async def _checker():
                while not self._cancel_token.is_set():
                    if self._queue.empty():
                        await asyncio.sleep(0)
                    else:
                        try:
                            task = dill.loads(self._queue.get())
                            fut = loop.create_task(_wrapper(task.coro, *task.args, **task.kwargs))
                            fut.add_done_callback(lambda f: self._queue.task_done())
                        except (EOFError, mp.managers.RemoteError):
                            self._cancel_token.set()
                        except Exception:
                            import traceback
                            print(mp.current_process().name, 'got an error:\n', traceback.format_exc(),
                                  file=sys.stderr, flush=True)

            def raise_keyboard():
                raise KeyboardInterrupt()

            def raise_exit():
                raise SystemExit()

            loop.add_signal_handler(SIGINT, raise_keyboard)
            loop.add_signal_handler(SIGTERM, raise_exit)

            try:
                loop.run_until_complete(_wrapper(_checker))
            except (KeyboardInterrupt, SystemExit):
                pass
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                tasks = []
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                    tasks.append(task)

                # print(mp.current_process().name, 'tasks to close', len(tasks), file=sys.stderr, flush=True)
                with suppress(asyncio.exceptions.CancelledError):
                    loop.run_until_complete(asyncio.gather(*tasks))

                loop.stop()
                loop.close()

    @property
    def n_tasks(self):
        if self._tasks_queue is None:
            return 0
        else:
            return self._tasks_queue._unfinished_tasks.get_value()

    def __init__(self):
        self._tasks_queue = _MP_CTX.JoinableQueue()
        self._tasks_queue.cancel_join_thread()
        self._cancel_token = _MP_CTX.Event()

        self._process = AsyncProcess.ProcessImpl(self._tasks_queue, self._cancel_token)
        # self._process.daemon = True

    def __del__(self):
        self.stop()
        self._process.join(0.1)
        self._process.terminate()
        self._process.join()
        self._process.close()

    def start(self):
        self._process.start()

    def stop(self):
        if self._cancel_token is not None:
            self._cancel_token.set()
            self._cancel_token = None
        if self._tasks_queue is not None:
            self._tasks_queue.close()
            self._tasks_queue = None

    def join(self, timeout=None):
        if self._process.is_alive():
            self._process.join(timeout)

    def add_task(self, task, args=None, kwargs=None):
        assert self._process.is_alive()
        assert self._tasks_queue is not None
        assert asyncio.iscoroutinefunction(task)

        if args is None:
            args = tuple()

        if kwargs is None:
            kwargs = dict()

        task = AsyncProcess._Task(coro=task, args=args, kwargs=kwargs)

        self._tasks_queue.put(dill.dumps(task))


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MetaSingleton, cls).__call__(*args, **kwargs)

        return cls.instance


class ProcessPool(metaclass=MetaSingleton):
    def __init__(self):
        self.ctx = _MP_CTX
        self.manager = None

        self._pool = None

        self._start()

    def __del__(self):
        self._stop()

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
            self.manager = self.ctx.Manager()

            import os
            self._pool = [AsyncProcess() for _ in range(os.cpu_count())]

            for t in self._pool:
                t.start()

    def _stop(self):
        if self._pool:
            for t in self._pool:
                t.stop()

            # dt = 1 / len(self._pool)
            for t in self._pool:
                t.join()

            self._pool.clear()

            self.manager.shutdown()

            self._pool = None


class _ParallelIterator:
    @staticmethod
    async def _parallel_process(map_func, input_queue, output_queue, cancel_token):
        import multiprocessing
        from .. import _dataset

        _map_func = dill.loads(map_func)

        if asyncio.iscoroutinefunction(map_func):
            map_func = _map_func
        else:
            async def _wrapper(*args, **kwargs):
                return _map_func(*args, **kwargs)

            map_func = _wrapper

        async def send_result(idx, next_flag, result):
            while output_queue.full():
                await asyncio.sleep(0)
            else:
                output_queue.put(dill.dumps((idx, next_flag, result)))

        while not cancel_token.is_set():
            if input_queue.empty():
                await asyncio.sleep(0)
            else:
                try:
                    idx, sample = dill.loads(input_queue.get())

                    try:
                        result = await map_func(*sample)
                        result = (True, result)
                    except Exception:
                        import sys
                        import traceback
                        print(multiprocessing.current_process().name, f'got an error for sample #{idx}:\n',
                              traceback.format_exc(),
                              file=sys.stderr, flush=True)

                        result = (False, RuntimeError(f'Sample #{idx}:\n' + traceback.format_exc()))

                    if isinstance(result[1], _dataset.Dataset):
                        async for item in result[1]:
                            await send_result(idx, False, (True, item))
                        await send_result(idx, True, (False, None))
                    else:
                        await send_result(idx, True, result)
                finally:
                    pass

    def __init__(self, session_id, source, map_func, n_workers, ordered, *, ignore_errors=False):
        self._pool = ProcessPool()

        self._stop_fetching = False

        self._n_workers = n_workers
        self._source_iter = aioitertools.enumerate(source)
        self._ignore_errors = ignore_errors

        map_func_dump = dill.dumps(map_func)

        # self._input_queue_n_tasks = _ParallelSession.manager.Semaphore(0)

        self._input_queue = self._pool.manager.Queue()
        self._output_queue = self._pool.manager.Queue()
        self._cancel_token = self._pool.manager.Event()

        self._fetch_next_idx = 0

        if ordered:
            def put_strategy(fetch_next_idx, sample_idx):
                return fetch_next_idx == sample_idx
        else:
            def put_strategy(fetch_next_idx, sample_idx):
                return True

        self._put_strategy = put_strategy

        for _ in range(n_workers):
            self._pool.submit(
                _ParallelIterator._parallel_process,
                args=(map_func_dump, self._input_queue, self._output_queue, self._cancel_token)
            )

        self._samples_in_process = 0
        self._result_bag = []

    def __del__(self):
        if self._pool is not None:
            try:
                if self._cancel_token is not None:
                    self._cancel_token.set()
            except FileNotFoundError:
                pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        while self._source_iter is not None or self._samples_in_process > 0:
            if self._source_iter is not None and not self._input_queue.full():
                try:
                    idx, sample = await aioitertools.next(self._source_iter)
                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    self._input_queue.put(dill.dumps((idx, sample)))
                    self._samples_in_process += 1
                except StopAsyncIteration:
                    self._source_iter = None

            while not self._output_queue.empty():
                self._result_bag.append(dill.loads(self._output_queue.get()))

            remove_list = []
            try:
                for i, (idx, next_flag, item) in enumerate(self._result_bag):
                    if self._put_strategy(self._fetch_next_idx, idx):
                        try:
                            (flag, result) = item
                            if flag:
                                return result
                            else:
                                if not self._ignore_errors and isinstance(result, Exception):
                                    raise result
                        finally:
                            if next_flag:
                                self._fetch_next_idx += 1
                                self._samples_in_process -= 1
                            remove_list.append(i)
            finally:
                for i in reversed(remove_list):
                    del self._result_bag[i]

            await asyncio.sleep(0)
        else:
            if self._cancel_token is not None:
                self._cancel_token.set()
                self._cancel_token = None
            raise StopAsyncIteration()


class MapDataOperation:
    def __init__(self, *, source, map_func, num_parallel_calls, ordered=True, ignore_errors=False):
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
