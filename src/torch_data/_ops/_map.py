

class _SerialIterator:
    def __init__(self, source, map_func):
        self._source_iter = iter(source)
        self._map_func = map_func

    def __iter__(self):
        return self

    def __next__(self):
        sample = next(self._source_iter)
        if not isinstance(sample, tuple):
            sample = (sample,)

        return self._map_func(*sample)


class _ParallelIterator:
    @staticmethod
    def _parallel_process(map_func, put_strategy,
                          input_queue, input_rlock,
                          output_queue, output_rlock,
                          fetched_idx, cancel_token):
        import dill
        import multiprocessing
        import time

        N_MAX_LOCALS = 2

        map_func = dill.loads(map_func)
        put_strategy = dill.loads(put_strategy)

        local_queue = []
        while not cancel_token.is_set():
            try:
                if len(local_queue) < N_MAX_LOCALS:
                    sample = None
                    if input_rlock.acquire(False):
                        try:
                            if not input_queue.empty():
                                sample = dill.loads(input_queue.get_nowait())
                        finally:
                            input_rlock.release()

                    if sample is not None:
                        i, sample = sample

                        result = map_func(*sample)

                        result = dill.dumps(result)

                        local_queue.append((i, result))

                i = 0
                while i < len(local_queue):
                    idx, result = local_queue[i]

                    if not output_rlock.acquire(False):
                        break
                    else:
                        try:
                            if not output_queue.full() and put_strategy(fetched_idx.value, idx):
                                output_queue.put_nowait(result)
                                del local_queue[i]
                                i -= 1
                        finally:
                            output_rlock.release()

                    i += 1
                else:
                    time.sleep(0)
            except Exception:
                import traceback
                print(multiprocessing.current_process().name, 'got an error:\n', traceback.format_exc())

    def __init__(self, source, map_func, put_strategy, n_workers):
        import dill
        import multiprocessing as mp

        self._stop_fetching = False

        self._n_workers = n_workers
        self._source_iter = iter(source)
        self._map_func = map_func
        self._put_strategy = put_strategy

        ctx = mp.get_context('spawn')

        self._qsize = n_workers * 3 // 2

        self._input_queue = ctx.Queue(maxsize=self._qsize)
        self._input_rlock = ctx.RLock()
        self._output_queue = ctx.Queue(maxsize=self._qsize)
        self._output_rlock = ctx.RLock()

        self._enumerator = -1
        self._fetched_idx = ctx.Value('i', -1)

        self._cancel_token = ctx.Event()
        self._cancel_token.clear()

        self._pool = [
            ctx.Process(target=_ParallelIterator._parallel_process,
                        args=(dill.dumps(self._map_func),
                              dill.dumps(self._put_strategy),
                              self._input_queue, self._input_rlock,
                              self._output_queue, self._output_rlock,
                              self._fetched_idx,
                              self._cancel_token))
            for _ in range(n_workers)
        ]

        for p in self._pool:
            p.start()

    def __del__(self):
        self._input_queue.cancel_join_thread()
        self._input_queue.close()

        self._cancel_token.set()

        self._output_queue.cancel_join_thread()

        for p in self._pool:
            try:
                p.join(0.1)
                p.close()
            except Exception:
                p.terminate()

    def __iter__(self):
        return self

    def __next__(self):
        import dill
        from concurrent.futures import Future

        while not self._stop_fetching:
            try:
                # fill the input queue untill it's full
                while not self._input_queue.full():
                    sample = next(self._source_iter)
                    self._enumerator += 1
                    i = self._enumerator

                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    self._input_queue.put_nowait(dill.dumps((i, sample)))

                # try to get a result
                if not self._output_queue.empty():
                    result = self._output_queue.get_nowait()
                    self._fetched_idx.value += 1
                    return dill.loads(result)

            except StopIteration:
                self._stop_fetching = True

        assert self._stop_fetching, ''

        while self._fetched_idx.value < self._enumerator:
            if not self._output_queue.empty():
                result = self._output_queue.get_nowait()
                self._fetched_idx.value += 1
                return dill.loads(result)
        else:
            raise StopIteration()


def _ordered_put_strategy(last_fetched_idx, sample_idx):
    return sample_idx == last_fetched_idx + 1


def _unordered_put_strategy(last_fetched_idx, sample_idx):
    return True


class MapDataOperation:
    def __init__(self, *, source, map_func, num_parallel_calls=None, ordered=True):
        if num_parallel_calls is None or num_parallel_calls == 0:
            self._get_iterator = lambda: _SerialIterator(source, map_func)
        else:
            if ordered:
                self._get_iterator = lambda: _ParallelIterator(
                    source, map_func, put_strategy=_ordered_put_strategy, n_workers=num_parallel_calls)
            else:
                self._get_iterator = lambda: _ParallelIterator(
                    source, map_func, put_strategy=_unordered_put_strategy, n_workers=num_parallel_calls)

    def __iter__(self):
        return self._get_iterator()
