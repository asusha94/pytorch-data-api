

class GeneratorIterator:
    def __init__(self, iterator_func):
        self._iter = iterator_func()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class GeneratorDataSource:
    def __init__(self, *, generator, args=None):
        if args is None:
            args = tuple()

        self._get_iterator = lambda: iter(generator(*args))

    def __iter__(self):
        return GeneratorIterator(self._get_iterator)
