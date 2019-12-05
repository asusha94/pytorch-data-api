
class DatasetIterator:
    def __init__(self, ds):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass


class Dataset:
    @staticmethod
    def from_generator(generator, args=None):
        pass

    @staticmethod
    def from_tensor_slices(tensors):
        pass

    @staticmethod
    def from_tensors(tensors):
        pass

    def __init__(self):
        pass

    def __iter__(self):
        return DatasetIterator(self)

    def filter(self, predicate):
        pass

    def map(self, map_func, num_parallel_calls=None):
        pass

    def shuffle(self, buffer_size, seed=None):
        pass

    def batch(self, batch_size):
        pass

    def unbatch(self):
        pass

    def window(self, size, shift=None, stride=1, drop_remainder=False):
        pass
