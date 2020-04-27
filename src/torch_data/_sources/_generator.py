

class _GeneratorIterator:
    def __init__(self, session_id, iterator):
        self._session_id = session_id
        self._iter = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class GeneratorDataSource:
    def __init__(self, *, generator, args=None):
        if args is None:
            args = tuple()

        self.__get_iterator = lambda sid: _GeneratorIterator(sid, iter(generator(*args)))

    def get_iter(self, session_id):
        return self.__get_iterator(session_id)
