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
        self.batch_size = 4
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
        opts = TestOptions()
        opts.learning_rate = .0001
        opts.fake_num_samples = 100
        opts.epochs_to_train = 100
        opts.num_hidden = 200
        with tf.Session() as session:
            dataset = datasets.FakeAdversarialDataset(opts)
            model = adversarial_nets.AdversarialNets(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):
                model.run_epoch()  

            samples = model.sample_space()
            print samples

        if SHOW_PLOTS:
            model.plot_results()
            plt.scatter(samples[:, 0], samples[:, 1], c=np.arange(len(samples)))
            plt.show()


if __name__ == '__main__':
    unittest.main()