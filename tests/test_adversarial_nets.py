import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets
import adversarial_nets

SHOW_PLOTS = True

class TestOptions(object):

    def __init__(self):
        self.fake_num_samples = 1000
        self.fake_val_ratio = 2
        self.batch_size = 8
        self.fake_input_dim = 2
        self.epochs_to_train = 20
        self.learning_rate = .1
        self.reg_scale = 1e-5
        self.run_validation = False
        self.num_hidden = 10
        self.z_dim = 1

class TestAdversarialNets(unittest.TestCase):

    def test_train_generator(self):
        """
        Make sure that the generator can achieve low loss, when 
        not training the discriminator.
        """
        opts = TestOptions()
        opts.fake_num_samples = 16
        with tf.Session() as session:
            dataset = datasets.FakeAdversarialDataset(opts)
            model = adversarial_nets.AdversarialNets(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):
                loss = model.train_generator()  
                losses.append(np.mean(loss))

        self.assertTrue(losses[-1] < 1e-2)

        if SHOW_PLOTS:
            plt.title("generator loss without discriminator training")
            plt.plot(losses)
            plt.show()

    def test_train_discriminator(self):
        """
        Make sure that the discriminator can achieve low loss, when 
        not training the generator.
        """
        opts = TestOptions()
        opts.fake_num_samples = 16
        with tf.Session() as session:
            dataset = datasets.FakeAdversarialDataset(opts)
            model = adversarial_nets.AdversarialNets(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):

                for bidx in range(opts.batch_size):
                    samples = np.tile([.5,.5], (opts.batch_size, 1))
                    model._dataset.add_generated_samples(samples)

                loss = model.train_discriminator()  
                losses.append(np.mean(loss))

        self.assertTrue(losses[-1] < 1e-2)

        if SHOW_PLOTS:
            plt.title("discriminator loss without generator training")
            plt.plot(losses)
            plt.show()

    def test_run_epoch(self):
        """
        Test with both training.
        """
        np.random.seed(1)
        opts = TestOptions()
        opts.learning_rate = .00001 # .001
        opts.train_ratio = 3 # 2
        opts.fake_num_samples = 128 # 128
        opts.epochs_to_train = 1000 # 500
        opts.num_hidden = 512 # 100
        opts.z_dim = 2 # 2
        opts.reg_scale = 0 # 0
        opts.dropout = 1 # 1
        with tf.Session() as session:
            dataset = datasets.FakeAdversarialDataset(opts)
            model = adversarial_nets.AdversarialNets(opts, session, dataset)

            losses = []
            for epoch in range(opts.epochs_to_train):
                model.run_epoch()  

            samples = model.sample_space()
            true_samples = model._dataset.data['X_train']
            print samples

        if SHOW_PLOTS:
            model.plot_results()
            plt.scatter(samples[:, 0], samples[:, 1], c='red')
            plt.scatter(true_samples[:, 0], true_samples[:, 1], c='blue')
            # plt.xlim([0,2])
            # plt.ylim([-1,1])
            plt.show()


if __name__ == '__main__':
    unittest.main()
