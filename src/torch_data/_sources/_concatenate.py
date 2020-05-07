import copy


class _ConcatenateIterator:
    _none = object()

    def __init__(self, session_id, dataset_iters):
        self._session_id = session_id
        self._dataset_iters = list(dataset_iters)

    def __iter__(self):
        return self

    def __next__(self):
        while self._dataset_iters:
            sample = next(self._dataset_iters[0], self._none)
            if sample is self._none:
                del self._dataset_iters[0]
            else:
                return copy.deepcopy(sample)
        else:
            raise StopIteration()


class ConcatenateDataSource:
    def __init__(self, *, dataset_sources):
        self._dataset_sources = dataset_sources

    def get_iter(self, session_id):
        return _ConcatenateIterator(session_id, [d.get_iter(session_id) for d in self._dataset_sources])
