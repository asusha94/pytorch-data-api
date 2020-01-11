

class _InterleaveIterator:
    _none = object()
    
    def __init__(self, datasets):
        self._datasets = list(datasets)
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self._datasets:
            try:
                sample = next(self._datasets[self._idx], self._none)
                if sample is self._none:
                    del self._datasets[self._idx]
                    self._idx -= 1
                else:
                    return sample
            finally: # cyclic
                self._idx += 1
                if self._idx >= len(self._datasets):
                    self._idx = 0
        else:
            raise StopIteration()


class InterleaveDataSource:
    def __init__(self, *datasets):
        self._datasets = datasets

    def __iter__(self):
        return _InterleaveIterator([iter(d) for d in self._datasets])
