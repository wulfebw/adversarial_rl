import numpy as np
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import learning_utils

class TestLearningUtils(unittest.TestCase):

    def test_batch_sample_with_temperature(self):
        sess = tf.InteractiveSession()
        batch = tf.convert_to_tensor([[0.,10.], [10.,0.]])
        idxs = learning_utils.batch_sample_with_temperature(batch)
        actual_idxs = sess.run(idxs)
        expected_idxs = [1,0]
        self.assertEquals(actual_idxs.tolist(), expected_idxs)

        batch = tf.convert_to_tensor([[0.,10.,30.], [6.,5.,3000.]])
        idxs = learning_utils.batch_sample_with_temperature(batch)
        actual_idxs = sess.run(idxs)
        expected_idxs = [2,2]
        self.assertEquals(actual_idxs.tolist(), expected_idxs)

if __name__ == '__main__':
    unittest.main()