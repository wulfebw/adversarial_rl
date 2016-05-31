import collections
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import learning_utils

class TestLearningUtils(unittest.TestCase):

    def test_batch_sample_with_temperature_basic(self):
        sess = tf.InteractiveSession()
        batch = tf.convert_to_tensor([[0.,10.], [10.,0.]])
        idxs = learning_utils.batch_sample_with_temperature(batch)
        actual_idxs, _ = sess.run(idxs)
        expected_idxs = [1,0]
        self.assertEquals(actual_idxs.tolist(), expected_idxs)

        batch = tf.convert_to_tensor([[0.,10.,30.], [6.,5.,3000.]])
        idxs = learning_utils.batch_sample_with_temperature(batch)
        actual_idxs, _ = sess.run(idxs)
        expected_idxs = [2,2]
        self.assertEquals(actual_idxs.tolist(), expected_idxs)

    def test_batch_sample_with_temperature_against_numpy(self):
        np.random.seed(1)
        num_samples = 100000
        K = 4
        batch = np.arange(K)
        batch = np.tile(batch, (num_samples, 1))
        exp_batch = np.exp(batch)
        sum_exp_batch = np.reshape(np.sum(exp_batch, axis=1), (-1, 1))
        softmax = exp_batch / sum_exp_batch
        np_indices = np.zeros(len(softmax))
        for idx, probs in enumerate(softmax):
            np_indices[idx] = np.argmax(np.random.multinomial(1, probs, 1))
        np_counts = collections.Counter()
        np_counts.update(np_indices)

        sess = tf.InteractiveSession()
        arr = tf.to_float(tf.convert_to_tensor(softmax))
        tf_indices, _ = sess.run(learning_utils.batch_sample_with_temperature(arr, temperature = 1.))
        tf_counts = collections.Counter()
        tf_counts.update(tf_indices)

        for x in range(K):
            print x
            print np_counts[x]
            print tf_counts[x]
            print "\n"
            plt.scatter(x, np_counts[x], c='blue')
            plt.scatter(x, tf_counts[x], c='red')
        plt.show()

if __name__ == '__main__':
    unittest.main()