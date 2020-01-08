import unittest

import torch_data


def map_fn(x):
    return x**2


class TestDataset(unittest.TestCase):
    def test_serial_map(self):
        ds = torch_data.Dataset.from_generator(range, args=(1000,))
        ds = ds.map(lambda x: x**2)

        for i, r in enumerate(ds):
            self.assertEqual(i**2, r)

        self.assertEqual(i, 999)

    def test_parallel_map_ordered(self):
        ds = torch_data.Dataset.from_generator(range, args=(1000,))
        ds = ds.map(lambda x: x**2, num_parallel_calls=4)

        for i, r in enumerate(ds):
            self.assertEqual(i**2, r)

        self.assertEqual(i, 999)

    def test_parallel_map_unordered(self):
        ds = torch_data.Dataset.from_generator(range, args=(1000,))
        ds = ds.map(lambda x: x**2, num_parallel_calls=4, ordered=False)

        sum_1 = 0
        sum_2 = 0
        for i, r in enumerate(ds):
            sum_1 += i**2
            sum_2 += r

        self.assertEqual(i, 999)
        self.assertEqual(sum_1, sum_2)


if __name__ == '__main__':
    unittest.main()
