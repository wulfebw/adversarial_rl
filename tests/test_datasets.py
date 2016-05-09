import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets

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

if __name__ == '__main__':
    unittest.main()