
class _SerialIterator:
    _none = object()

    def __init__(self, source, predicate, expand_args):
        self._source_iter = iter(source)
        self._predicate = predicate
        self._expand_args = expand_args

    def __iter__(self):
        return self

    def __next__(self):
        while self._source_iter is not None:
            sample = next(self._source_iter, self._none)

            if sample is self._none:
                self._source_iter = None
            else:
                if not self._expand_args:
                    val = self._predicate(sample)
                else:
                    if not isinstance(sample, tuple):
                        sample = (sample,)
                    val = self._predicate(*sample)

                if val:
                    return sample
        else:
            raise StopIteration()


class FilterDataOperation:
    def __init__(self, *, source, predicate, expand_args):
        self._source = source
        self._predicate = predicate
        self._expand_args = expand_args

    def get_iter(self, session_id):
        return _SerialIterator(self._source.get_iter(session_id), self._predicate, self._expand_args)
