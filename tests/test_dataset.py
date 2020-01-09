import unittest

import torch_data


class TestDataset(unittest.TestCase):
    def test_from_generator(self):
        ds = torch_data.Dataset.from_generator(range, args=(1000,))
        for i, r in enumerate(ds):
            self.assertEqual(i, r)

    def test_from_tensor_slices(self):
        self.assertRaises(AssertionError, torch_data.Dataset.from_tensor_slices)
        self.assertRaises(AssertionError, torch_data.Dataset.from_tensor_slices, [1, 2], [2], tensors=([1], [2]))

        tensor1 = list(range(1000))
        tensor2 = [str(i) + 'i' for i in range(1000)]

        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        for i, r in enumerate(ds):
            self.assertEqual(tensor1[i], r[0])
            self.assertEqual(tensor2[i], r[1])

        self.assertEqual(i, 999)

        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2[:500])
        for i, r in enumerate(ds):
            self.assertEqual(tensor1[i], r[0])
            self.assertEqual(tensor2[i], r[1])

        self.assertEqual(i, 499)

        ds = torch_data.Dataset.from_tensor_slices(tensor1[:250], tensor2)
        for i, r in enumerate(ds):
            self.assertEqual(tensor1[i], r[0])
            self.assertEqual(tensor2[i], r[1])

        self.assertEqual(i, 249)

    def test_from_tensors(self):
        self.assertRaises(AssertionError, torch_data.Dataset.from_tensors)
        self.assertRaises(AssertionError, torch_data.Dataset.from_tensors, [1, 2], [2], tensors=([1], [2]))

        tensor1 = list(range(1000))
        tensor2 = [str(i) + 'i' for i in range(1000)]

        ds = torch_data.Dataset.from_tensors(tensor1, tensor2)
        ds_iter = iter(ds)

        item = next(ds_iter)

        self.assertEqual(item[0], tensor1)
        self.assertEqual(item[1], tensor2)

        self.assertRaises(StopIteration, next, ds_iter)

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
