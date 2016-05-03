import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets

class TestOptions(object):
    pass

class TestFakeAdversarialDataset(unittest.TestCase):

    def test_make_fake_dataset(self):
        opts = TestOptions()
        opts.fake_num_samples = 10
        opts.fake_val_ratio = 2
        opts.batch_size = 4
        d = datasets.FakeAdversarialDataset(opts)
        actual = d.data['X_train'] 
        expected = np.tile([1,0], (10, 1))
        self.assertEquals(actual.tolist(), expected.tolist())

    def test_next_batch(self):
        opts = TestOptions()
        opts.fake_num_samples = 4
        opts.fake_val_ratio = 2
        opts.batch_size = 4
        d = datasets.FakeAdversarialDataset(opts)
        for idx in range(opts.fake_num_samples):
            d.add_generated_sample([.5,.5])
        X, y = d.next_batch()
        self.assertEquals(len(X), opts.batch_size)
        self.assertEquals(len(y), opts.batch_size)
        X, y = d.next_batch()
        self.assertRaises(AssertionError, d.next_batch)

if __name__ == '__main__':
    unittest.main()