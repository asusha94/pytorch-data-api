

class _InterleaveIterator:
    _none = object()

    def __init__(self, session_id, dataset_iters, drop_tails):
        self._session_id = session_id
        self._dataset_iters = dataset_iters

        self._drop_tails = drop_tails
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self._dataset_iters:
            try:
                sample = next(self._dataset_iters[self._idx], self._none)
                if sample is self._none:
                    if self._drop_tails:
                        self._dataset_iters.clear()
                    else:
                        del self._dataset_iters[self._idx]
                        self._idx -= 1
                else:
                    return sample
            finally:  # cyclic
                self._idx += 1
                if self._idx >= len(self._dataset_iters):
                    self._idx = 0
        else:
            raise StopIteration()


class InterleaveDataSource:
    def __init__(self, *, dataset_sources, drop_tails):
        self._dataset_sources = dataset_sources
        self._drop_tails = drop_tails

    def get_iter(self, session_id):
        sources = [d.get_iter(session_id) for d in self._dataset_sources]
        return _InterleaveIterator(session_id, sources, self._drop_tails)
