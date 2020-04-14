

class _InterleaveIterator:
    _none = object()

    def __init__(self, datasets, drop_tails):
        self._datasets = list(datasets)
        self._drop_tails = drop_tails
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self._datasets:
            try:
                sample = next(self._datasets[self._idx], self._none)
                if sample is self._none:
                    if self._drop_tails:
                        self._datasets.clear()
                    else:
                        del self._datasets[self._idx]
                        self._idx -= 1
                else:
                    return sample
            finally:  # cyclic
                self._idx += 1
                if self._idx >= len(self._datasets):
                    self._idx = 0
        else:
            raise StopIteration()


class InterleaveDataSource:
    def __init__(self, *datasets, drop_tails):
        self._datasets = datasets
        self._drop_tails = drop_tails

    def __iter__(self):
        return _InterleaveIterator([iter(d) for d in self._datasets], self._drop_tails)
