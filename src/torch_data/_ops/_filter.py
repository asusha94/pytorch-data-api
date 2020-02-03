
class _SerialIterator:
    _none = object()

    def __init__(self, source, predicate, expand_args):
        self._source_iter = iter(source)
        self._predicate = predicate
        self._expand_args = expand_args

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            sample = next(self._source_iter, self._none)

            if sample is self._none:
                raise StopIteration()
            else:
                if not self._expand_args:
                    val = self._predicate(sample)
                else:
                    if not isinstance(sample, tuple):
                        sample = (sample,)
                    val = self._predicate(*sample)

                if val:
                    return sample


class FilterDataOperation:
    def __init__(self, *, source, predicate, expand_args):
        self._source = source
        self._predicate = predicate
        self._expand_args = expand_args

    def __iter__(self):
        return _SerialIterator(iter(self._source), self._predicate, self._expand_args)
