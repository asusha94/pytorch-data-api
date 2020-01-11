

class _GeneratorIterator:
    def __init__(self, iterator):
        self._iter = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class GeneratorDataSource:
    def __init__(self, *, generator, args=None):
        if args is None:
            args = tuple()

        self.__get_iterator = lambda: _GeneratorIterator(iter(generator(*args)))

    def __iter__(self):
        return self.__get_iterator()
