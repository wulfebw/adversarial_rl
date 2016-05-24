import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets
import discrete_rgan

SHOW_PLOTS = True

class TestOptions(object):

    def __init__(self):
        self.learning_rate = .01 #.001
        self.input_dim = 3
        self.embed_dim = 30
        self.batch_size = 6
        self.train_ratio = 2 # 2
        self.sequence_length = 2
        self.epochs_to_train = 20 # 500
        self.num_hidden = 32 # 100
        self.z_dim = 2 # 2
        self.reg_scale = 0 # 0
        self.dropout = 1 # 1
        self.z_lim = 1
        self.dataset_name = 'alphabet'
        self.discrete = True
        self.num_samples = 10

class TestRecurrentDiscreteGenerativeAdversarialNetwork(unittest.TestCase):

    def test_train_generator(self):
        """
        Make sure that the generator can achieve low loss, when 
        not training the discriminator.
        """
        opts = TestOptions()
        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):
                loss = model.train_generator()  
                losses.append(np.mean(loss))

        if SHOW_PLOTS:
            plt.title("generator loss without discriminator training")
            plt.plot(losses)
            plt.show()

        self.assertTrue(losses[-1] < 1e-2)
        final_dataset = model.dataset.data['generated_samples']
        

    def test_train_discriminator(self):
        """
        Make sure that the discriminator can achieve low loss, when 
        not training the generator.
        """
        opts = TestOptions()
        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):

                for bidx in range(opts.batch_size):
                    samples = np.zeros((opts.batch_size, opts.sequence_length))
                    model.dataset.add_generated_samples(samples)

                loss = model.train_discriminator()  
                losses.append(np.mean(loss))

        if SHOW_PLOTS:
            plt.title("discriminator loss without generator training")
            plt.plot(losses)
            plt.show()

        self.assertTrue(losses[-1] < 1e-2)

    def test_run_epoch(self):
        """
        Test with both training.
        """
        opts = TestOptions()    
        opts.learning_rate = .001
        opts.epochs_to_train = 50
        opts.num_hidden = 16
        opts.embed_dim = 4
        opts.dropout = 1 

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            # saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            losses = []
            
            TRAIN = True
            if TRAIN == True:
                for epoch in range(opts.epochs_to_train):
                    model.run_epoch()  

                    if epoch % 10000 == 0:
                        saver.save(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            if opts.discrete:
                samples = model.discrete_sample_space()
                samples = dataset.decode_dataset(samples)
            else:
                samples = model.sample_space()

            true_samples = dataset.data['X_train']
            true_samples = dataset.decode_dataset(true_samples)

            total = float(len(samples))
            in_real_data_count = 0
            for sample in samples:
                if sample in true_samples:
                    in_real_data_count += 1
            print "example generated data: {}".format(samples[:5])
            print "total samples: {}".format(total)
            print "fake samples in dataset: {}".format(in_real_data_count)
            print "percent in real dataset: {}".format(100 * in_real_data_count / total)

            if SHOW_PLOTS:
                model.plot_results()

if __name__ == '__main__':
    unittest.main()
