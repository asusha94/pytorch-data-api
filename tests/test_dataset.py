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

    def test_concatenate(self):
        self.assertRaises(AssertionError, torch_data.Dataset.concatenate)
        self.assertRaises(AssertionError, torch_data.Dataset.concatenate, torch_data.Dataset())
        self.assertRaises(AssertionError, torch_data.Dataset.concatenate, [1, 2], torch_data.Dataset())
        self.assertRaises(AssertionError, torch_data.Dataset.concatenate, torch_data.Dataset(), [1, 2], torch_data.Dataset())

        ds = torch_data.Dataset.concatenate(torch_data.Dataset(), torch_data.Dataset(), torch_data.Dataset())
        self.assertRaises(StopIteration, next, iter(ds))

        ds = torch_data.Dataset.concatenate(
            torch_data.Dataset.from_tensor_slices([1, 2, 3, 4]),
            torch_data.Dataset.from_tensor_slices(['1', '2', '3'])
        )

        out = []
        for i, r in enumerate(ds):
            out.append(r)
        
        self.assertEqual(i, 6)
        self.assertEqual(tuple(out), (1, 2, 3, 4, '1', '2', '3'))

    def test_interleave(self):
        self.assertRaises(AssertionError, torch_data.Dataset.interleave)
        self.assertRaises(AssertionError, torch_data.Dataset.interleave, torch_data.Dataset())
        self.assertRaises(AssertionError, torch_data.Dataset.interleave, [1, 2], torch_data.Dataset())
        self.assertRaises(AssertionError, torch_data.Dataset.interleave, torch_data.Dataset(), [1, 2], torch_data.Dataset())

        ds = torch_data.Dataset.interleave(torch_data.Dataset(), torch_data.Dataset(), torch_data.Dataset())
        self.assertRaises(StopIteration, next, iter(ds))

        ds = torch_data.Dataset.interleave(
            torch_data.Dataset.from_tensor_slices([1, 2, 3, 4]),
            torch_data.Dataset.from_tensor_slices(['1', '2', '3'])
        )

        out = []
        for i, r in enumerate(ds):
            out.append(r)
        
        self.assertEqual(i, 6)
        self.assertEqual(tuple(out), (1, '1', 2, '2', 3, '3', 4))

    def test_serial_map(self):
        ds = torch_data.Dataset.from_generator(range, args=(100,))
        ds = ds.map(lambda x: x**2)

        for i, r in enumerate(ds):
            self.assertEqual(i**2, r)

        self.assertEqual(i, 99)

    def test_parallel_map_ordered(self):
        ds = torch_data.Dataset.from_generator(range, args=(100,))
        ds = ds.map(lambda x: x**2, num_parallel_calls=3)

        for i, r in enumerate(ds):
            self.assertEqual(i**2, r)

        self.assertEqual(i, 99)

    def test_parallel_map_unordered(self):
        ds = torch_data.Dataset.from_generator(range, args=(100,))
        ds = ds.map(lambda x: x**2, num_parallel_calls=3, ordered=False)

        sum_1 = 0
        sum_2 = 0
        for i, r in enumerate(ds):
            sum_1 += i**2
            sum_2 += r

        self.assertEqual(i, 99)
        self.assertEqual(sum_1, sum_2)

    def test_shuffle(self):
        tensor1 = list(range(100))
        tensor2 = [str(i) + 'i' for i in range(100)]

        # batch_size=1
        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        self.assertRaises(AssertionError, ds.shuffle, 1)
        
        ds = ds.shuffle(5)

        out1 = []
        out2 = []
        for i, r in enumerate(ds):
            out1.append(r[0])
            out2.append(r[1])

        self.assertEqual(len(tensor1), len(out1))
        self.assertNotEqual(tuple(tensor1), tuple(out1))
        self.assertEqual(len(tensor2), len(out2))
        self.assertNotEqual(tuple(tensor2), tuple(out2))

    def test_batch(self):
        tensor1 = list(range(100))
        tensor2 = [str(i) + 'i' for i in range(100)]

        # batch_size=1
        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        ds = ds.batch(1)

        for i, r in enumerate(ds):
            # self.assertTrue(isinstance(r[0], list))
            self.assertEqual(len(r[0]), 1)

            self.assertTrue(isinstance(r[1], list))
            self.assertEqual(len(r[1]), 1)

            self.assertEqual(tensor1[i], r[0][0])
            self.assertEqual(tensor2[i], r[1][0])

        self.assertEqual(i, 99)

        # batch_size=2
        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        ds = ds.batch(2)

        for i, r in enumerate(ds):
            # self.assertTrue(isinstance(r[0], list))
            self.assertEqual(len(r[0]), 2)

            self.assertTrue(isinstance(r[1], list))
            self.assertEqual(len(r[1]), 2)

            self.assertEqual(tensor1[i * 2], r[0][0])
            self.assertEqual(tensor1[i * 2 + 1], r[0][1])

            self.assertEqual(tensor2[i * 2], r[1][0])
            self.assertEqual(tensor2[i * 2 + 1], r[1][1])

        self.assertEqual(i, 49)

        # batch_size=3
        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        ds = ds.batch(3)

        for i, r in enumerate(ds):
            # self.assertTrue(isinstance(r[0], list))
            self.assertEqual(len(r[0]), 3)

            self.assertTrue(isinstance(r[1], list))
            self.assertEqual(len(r[1]), 3)

            self.assertEqual(tensor1[i * 3], r[0][0])
            self.assertEqual(tensor1[i * 3 + 1], r[0][1])
            self.assertEqual(tensor1[i * 3 + 2], r[0][2])

            self.assertEqual(tensor2[i * 3], r[1][0])
            self.assertEqual(tensor2[i * 3 + 1], r[1][1])
            self.assertEqual(tensor2[i * 3 + 2], r[1][2])

        self.assertEqual(i, 32)

        # batch_size=3
        ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
        ds = ds.batch(3, drop_last=False)

        for i, r in enumerate(ds):
            if i == 33:
                # self.assertTrue(isinstance(r[0], list))
                self.assertEqual(len(r[0]), 3)

                self.assertTrue(isinstance(r[1], list))
                self.assertEqual(len(r[1]), 3)

                self.assertEqual(tensor1[i * 3], r[0][0])
                # self.assertEqual(None, r[0][1])
                # self.assertEqual(None, r[0][2])

                self.assertEqual(tensor2[i * 3], r[1][0])
                self.assertEqual(None, r[1][1])
                self.assertEqual(None, r[1][2])
            else:
                # self.assertTrue(isinstance(r[0], list))
                self.assertEqual(len(r[0]), 3)

                self.assertTrue(isinstance(r[1], list))
                self.assertEqual(len(r[1]), 3)

                self.assertEqual(tensor1[i * 3], r[0][0])
                self.assertEqual(tensor1[i * 3 + 1], r[0][1])
                self.assertEqual(tensor1[i * 3 + 2], r[0][2])

                self.assertEqual(tensor2[i * 3], r[1][0])
                self.assertEqual(tensor2[i * 3 + 1], r[1][1])
                self.assertEqual(tensor2[i * 3 + 2], r[1][2])

        self.assertEqual(i, 33)

        # numpy tensors
        try:
            import numpy as np
            tensor1 = np.array(tensor1)
            tensor2 = np.arange(len(tensor2) * 2).reshape(-1, 2)

            ds = torch_data.Dataset.from_tensor_slices(tensor1, tensor2)
            ds = ds.batch(3, drop_last=True)

            for i, r in enumerate(ds):
                self.assertTrue(isinstance(r[0], np.ndarray))
                self.assertEqual(np.shape(r[0]), (3,))

                self.assertTrue(isinstance(r[1], np.ndarray))
                self.assertEqual(np.shape(r[1]), (3, 2))

                self.assertEqual(tensor1[i * 3], r[0][0])
                self.assertEqual(tensor1[i * 3 + 1], r[0][1])
                self.assertEqual(tensor1[i * 3 + 2], r[0][2])

                self.assertTrue(np.all(tensor2[i * 3] == r[1][0]))
                self.assertTrue(np.all(tensor2[i * 3 + 1] == r[1][1]))
                self.assertTrue(np.all(tensor2[i * 3 + 2] == r[1][2]))

            self.assertEqual(i, 32)
        except (ImportError, ModuleNotFoundError):
            pass


if __name__ == '__main__':
    unittest.main()
