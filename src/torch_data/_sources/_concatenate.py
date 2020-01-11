
class _ConcatenateIterator:
    _none = object()

    def __init__(self, datasets):
        self._datasets = list(datasets)

    def __iter__(self):
        return self

    def __next__(self):
        while self._datasets:
            sample = next(self._datasets[0], self._none)
            if sample is self._none:
                del self._datasets[0]
            else:
                return sample
        else:
            raise StopIteration()


class ConcatenateDataSource:
    def __init__(self, *datasets):
        self._datasets = datasets

    def __iter__(self):
        return _ConcatenateIterator([iter(d) for d in self._datasets])
