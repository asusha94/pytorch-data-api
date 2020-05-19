import threading
import queue


class _PrefetchIterator:
    _none = object()

    @staticmethod
    def _prefetch_fn(output_queue, source_iter, cancel_token):
        while not cancel_token.is_set():
            sample = next(source_iter, _PrefetchIterator._none)
            output_queue.put(sample)
            if sample is _PrefetchIterator._none:
                break

    def __init__(self, source_iter, buffer_size):
        self._buffer = queue.Queue(buffer_size)
        self._cancel_token = threading.Event()

        self._thread = threading.Thread(target=_PrefetchIterator._prefetch_fn,
                                        args=(self._buffer, source_iter, self._cancel_token))
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        if self._thread.is_alive():
            self._cancel_token.set()
            self._thread.join(1e-1)

    def __iter__(self):
        return self

    def __next__(self):
        if self._buffer is None:
            raise StopIteration
        else:
            sample = self._buffer.get()
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
        return _PrefetchIterator(self._source.get_iter(session_id), self._buffer_size)
