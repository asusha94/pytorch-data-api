

class _SerialIterator:
    _none = object()

    def __init__(self, source, map_func, *, ignore_errors=False):
        self._source_iter = source
        self._map_func = map_func

        self._ignore_errors = ignore_errors

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self._source_iter, self._none)

        if sample is self._none:
            raise StopIteration()
        else:
            if not isinstance(sample, tuple):
                sample = (sample,)

            try:
                return self._map_func(*sample)
            except Exception:
                if not self._ignore_errors:
                    raise
                else:
                    import traceback
                    traceback.print_exc()

                    return None


class _ParallelSession:
    class Pool:
        @property
        def uid(self):
            return self._uid

        def __init__(self, uid):
            import multiprocessing as mp

            self._ctx = mp.get_context('spawn')

            self._uid = uid
            self._counter = 0

            self._pool = None

        def submit(self, task, args=None, kwargs=None):
            assert self._pool is not None, 'The Pool has been disposed'

            if args is None:
                args = tuple()

            if kwargs is None:
                kwargs = dict()

            return self._pool.apply_async(task, args=args, kwds=kwargs)

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


class _ParallelIterator:
    _none = object()

    @staticmethod
    def _parallel_process(map_func, sample, ignore_errors=False):
        import dill
        import multiprocessing

        map_func = dill.loads(map_func)
        sample = dill.loads(sample)

        try:
            result = map_func(*sample)
            return dill.dumps(result)
        except Exception:
            import traceback
            print(multiprocessing.current_process().name, 'got an error:\n', traceback.format_exc())

            if not ignore_errors:
                raise

    def __init__(self, session_id, source, map_func, n_workers, ordered, *, ignore_errors=False):
        import dill

        self._pool = _ParallelSession.get(session_id)

        self._stop_fetching = False

        self._n_workers = n_workers
        self._source_iter = source

        self._map_func_dump = dill.dumps(map_func)

        self._ignore_errors = ignore_errors

        self._source_disposed = False

        self._queue = []

        self._ordered = ordered

    def __del__(self):
        if self._pool is not None:
            _ParallelSession.release(self._pool)
            self._pool = None

    def __iter__(self):
        return self

    def __next__(self):
        import dill
        import time

        if self._pool is None:
            raise StopIteration()
        else:
            while not self._source_disposed and len(self._queue) < self._n_workers:
                sample = next(self._source_iter, self._none)
                if sample is self._none:
                    self._source_disposed = True
                else:
                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    result = self._pool.submit(self._parallel_process,
                                               args=(self._map_func_dump, dill.dumps(sample), self._ignore_errors))
                    self._queue.append(result)

            if not self._queue:
                _ParallelSession.release(self._pool)
                self._pool = None
                raise StopIteration()
            else:
                result_idx = None
                if self._ordered:
                    while not self._queue[0].ready():
                        time.sleep(1e-9)
                    else:
                        result_idx = 0
                else:
                    while result_idx is None:
                        ready = [i for (i, result) in enumerate(self._queue) if result.ready()]
                        if ready:
                            result_idx = ready[0]
                        else:
                            time.sleep(1e-9)

                result = self._queue[result_idx].get()
                del self._queue[result_idx]
                return dill.loads(result)


class MapDataOperation:
    def __init__(self, *, source, map_func, num_parallel_calls=None, ordered=True, ignore_errors=False):
        if num_parallel_calls is None or num_parallel_calls == 0:
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
