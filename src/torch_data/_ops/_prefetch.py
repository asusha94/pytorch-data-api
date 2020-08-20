import threading
import queue


class _ParallelSession:
    class Pool:
        @property
        def uid(self):
            return self._uid

        def __init__(self, uid):
            import multiprocessing.dummy as mp

            self._ctx = mp

            self._uid = uid
            self._counter = 0

            self._pool = None

        def submit(self, task, args=None, kwargs=None, callback=None):
            assert self._pool is not None, 'The Pool has been disposed'

            if args is None:
                args = tuple()

            if kwargs is None:
                kwargs = dict()

            return self._pool.apply_async(task, args=args, kwds=kwargs, callback=callback)

        def _start(self):
            if not self._pool:
                import os
                self._pool = self._ctx.Pool(os.cpu_count())

        def _stop(self):
            if self._pool:
                self._pool.close()

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

    @staticmethod
    def get(uid):
        pool = _ParallelSession._pools.get(uid)
        if pool is None:
            pool = _ParallelSession.Pool(uid)
            _ParallelSession._pools[uid] = pool

        pool._increment_ref()

        return pool

    @staticmethod
    def release(pool):
        if pool._decrement_ref():
            del _ParallelSession._pools[pool.uid]


class _PrefetchIterator:
    _none = object()

    @staticmethod
    def _prefetch_fn(output_queue, source_iter, cancel_token):
        if cancel_token.is_set():
            return False
        else:
            sample = next(source_iter, _PrefetchIterator._none)
            output_queue.put(sample)

            if sample is _PrefetchIterator._none:
                return False
            else:
                return True

    def __init__(self, session_id, source_iter, buffer_size):
        buffer = queue.Queue(buffer_size)
        self._buffer = buffer

        cancel_token = threading.Event()
        self._cancel_token = cancel_token
        pool = _ParallelSession.get(session_id)
        self._pool = pool

        def submit_iteration(f=True):
            if not f:
                return

            pool.submit(_PrefetchIterator._prefetch_fn,
                        args=(buffer, source_iter, cancel_token),
                        callback=submit_iteration)

        submit_iteration()

    def __del__(self):
        if self._pool is not None:
            self._cancel_token.set()
            _ParallelSession.release(self._pool)
            self._pool = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._buffer is None:
            raise StopIteration
        else:
            sample = self._buffer.get()
            self._buffer.task_done()

            if sample is self._none:
                self._buffer = None
                self._cancel_token.set()
                raise StopIteration
            else:
                return sample


class PrefetchDataOperation:
    def __init__(self, *, source, buffer_size):
        self._source = source
        self._buffer_size = buffer_size

    def get_iter(self, session_id):
        return _PrefetchIterator(session_id, self._source.get_iter(session_id), self._buffer_size)
