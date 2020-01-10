
class _ShuffleIterator:
    _none = object()

    def __iter__(self):
        return self

    def __next__(self):
        pass


class ShuffleDataOperation:
    def __init__(self, *, source, buffer_size, seed=None):
        pass

    def __iter__(self):
        pass
