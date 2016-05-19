import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets

SHOW_PLOTS = True

class TestOptions(object):

    def __init__(self):
        self.fake_num_samples = 1000
        self.fake_val_ratio = 2
        self.batch_size = 4
        self.fake_input_dim = 2

class TestFakeAdversarialDataset(unittest.TestCase):

    def test_make_fake_dataset(self):
        opts = TestOptions()
        d = datasets.FakeAdversarialDataset(opts)
        actual = d.data['X_train'] 
        expected = np.tile([1,0], (opts.fake_num_samples, 1))
        self.assertEquals(actual.tolist(), expected.tolist())

    def test_next_batch(self):
        opts = TestOptions()
        input_dim = opts.fake_input_dim
        d = datasets.FakeAdversarialDataset(opts)
        fake_batch = np.arange((opts.batch_size * input_dim * 2)).reshape(-1, input_dim)
        d.add_generated_samples(fake_batch[:len(fake_batch) / 2])
        d.add_generated_samples(fake_batch[len(fake_batch) / 2:])
        actual_batches = [(x,y) for x, y in d.next_batch()]
        self.assertEquals(len(actual_batches), opts.batch_size)

class TestFakeRecurrentAdversarialDataset(unittest.TestCase):

    def test_make_fake_dataset(self):
        opts = TestOptions()
        opts.sequence_length = 4
        opts.num_samples = 32
        opts.batch_size = 4
        opts.input_dim = 2
        d = datasets.FakeRecurrentAdversarialDataset(opts)

        z = d.data['X_train']
        for sidx in range(opts.num_samples):
            plt.plot(z[sidx, :, 0], z[sidx, :, 1])

        if SHOW_PLOTS:
            plt.show()

        self.assertEquals(len(d.data['X_train']), opts.num_samples)
        self.assertEquals(len(d.data['X_train'][0]), opts.sequence_length)

    def test_add_generated_samples(self):
        opts = TestOptions()
        opts.sequence_length = 2
        opts.num_samples = 4
        opts.batch_size = 2
        opts.input_dim = 2
        d = datasets.FakeRecurrentAdversarialDataset(opts)

        fake_batch = np.arange((opts.batch_size * opts.sequence_length * opts.input_dim))
        fake_batch = fake_batch.reshape(opts.batch_size, opts.sequence_length, opts.input_dim)
        d.add_generated_samples(fake_batch)
        d.add_generated_samples(fake_batch)
        fake_x = d.data['generated_samples']

        self.assertEquals(len(fake_x), opts.batch_size * 2)
        self.assertEquals(len(fake_x[0]), opts.sequence_length)

    def test_next_batch(self):
        opts = TestOptions()
        opts.sequence_length = 4
        opts.num_samples = 32
        opts.batch_size = 4
        opts.input_dim = 2
        d = datasets.FakeRecurrentAdversarialDataset(opts)

        fake_batch = np.arange((opts.batch_size * opts.sequence_length * opts.input_dim))
        fake_batch = fake_batch.reshape(opts.batch_size, opts.sequence_length, opts.input_dim)
        d.add_generated_samples(fake_batch)
        d.add_generated_samples(fake_batch)
        for (x, y) in d.next_batch():
            self.assertEquals(x.shape, (opts.batch_size, opts.sequence_length, opts.input_dim))
            self.assertEquals(y.shape, (opts.batch_size, 1))

if __name__ == '__main__':
    unittest.main()