

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
    def _parallel_process(map_func, input_queue, output_queue, cancel_token):
        import dill
        import multiprocessing

        map_func = dill.loads(map_func)

        while not cancel_token.is_set():
            sample = input_queue.get()
            print(multiprocessing.current_process().name, sample)
            if sample is None:
                break

            result = map_func(*sample)

            output_queue.put(result)

    def __init__(self, source, map_func, n_workers):
        import dill
        import multiprocessing as mp

        self._stop_fetching = False
        self._in_progress = 0

        self._n_workers = n_workers
        self._source_iter = iter(source)
        self._map_func = map_func

        ctx = mp.get_context('spawn')

        self._qsize = n_workers * 3 // 2

        self._input_queue = ctx.Queue(maxsize=self._qsize)
        self._output_queue = ctx.Queue(maxsize=self._qsize)
        self._cancel_token = ctx.Event()
        self._cancel_token.clear()

        self._pool = [
            ctx.Process(target=_ParallelIterator._parallel_process,
                        args=(dill.dumps(self._map_func), self._input_queue, self._output_queue, self._cancel_token))
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
        while not self._stop_fetching:
            try:
                # fill the input queue untill it's full
                while not self._input_queue.full():
                    sample = next(self._source_iter)
                    print('__next__', sample)

                    if not isinstance(sample, tuple):
                        sample = (sample,)

                    self._input_queue.put(sample)
                    self._in_progress += 1

                # try to get a result
                if not self._output_queue.empty():
                    result = self._output_queue.get()
                    self._in_progress -= 1
                    return result

            except StopIteration:
                self._stop_fetching = True

        assert self._stop_fetching, ''

        while self._in_progress > 0:
            if not self._output_queue.empty():
                result = self._output_queue.get()
                self._in_progress -= 1
                return result
        else:
            raise StopIteration()


class MapDataOperation:
    def __init__(self, *, source, map_func, num_parallel_calls=None, ordered=True):
        if num_parallel_calls is None or num_parallel_calls == 0:
            self._get_iterator = lambda: _SerialIterator(source, map_func)
        else:
            if ordered:  # TODO: implement ordered iterator
                self._get_iterator = lambda: _ParallelIterator(source, map_func, n_workers=num_parallel_calls)
            else:
                self._get_iterator = lambda: _ParallelIterator(source, map_func, n_workers=num_parallel_calls)

    def __iter__(self):
        return self._get_iterator()
