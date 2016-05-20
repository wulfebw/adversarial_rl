import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))

import datasets
import rgan

SHOW_PLOTS = True

class TestOptions(object):

    def __init__(self):
        self.num_samples = 32
        self.batch_size = 8
        self.input_dim = 2
        self.sequence_length = 4
        self.epochs_to_train = 5
        self.learning_rate = .1
        self.reg_scale = 1e-5
        self.dropout = 1
        self.run_validation = False
        self.num_hidden = 10
        self.train_ratio = 2
        self.z_dim = 2
        self.dataset_name = 'sine'

class TestRecurrentGenerativeAdversarialNetwork(unittest.TestCase):

    def test_train_generator(self):
        """
        Make sure that the generator can achieve low loss, when 
        not training the discriminator.
        """
        opts = TestOptions()
        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = rgan.RecurrentGenerativeAdversarialNetwork(opts, session, dataset)

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
        expected_num_samples = opts.epochs_to_train * opts.num_samples
        self.assertEquals(expected_num_samples, len(final_dataset))

    def test_train_discriminator(self):
        """
        Make sure that the discriminator can achieve low loss, when 
        not training the generator.
        """
        opts = TestOptions()
        opts.num_samples = 16
        opts.epochs_to_train = 10
        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = rgan.RecurrentGenerativeAdversarialNetwork(opts, session, dataset)

            # train only the generator
            losses = []
            for epoch in range(opts.epochs_to_train):

                for bidx in range(opts.batch_size):
                    samples = np.tile([.5,.5], (opts.batch_size, opts.sequence_length, 1))
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
        opts.learning_rate = .00001 #.001
        opts.batch_size = 8
        opts.train_ratio = 4 # 2
        opts.num_samples = 16 # 128
        opts.sequence_length = 20
        opts.epochs_to_train = 2000 # 500
        opts.num_hidden = 128 # 100
        opts.z_dim = 3 # 2
        opts.reg_scale = 0 # 0
        opts.dropout = .9 # 1
        opts.dataset_name = 'sine'

        

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = rgan.RecurrentGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            losses = []
            
            TRAIN = False
            if TRAIN == True:
                for epoch in range(opts.epochs_to_train):
                    model.run_epoch()  

                    if epoch % 50 == 0:
                        saver.save(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            samples = model.sample_space()
            true_samples = model.dataset.data['X_train']
            print samples[0:2]

        with tf.Session() as session:
            if SHOW_PLOTS:
                saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))
                model.plot_results()

                fig, ax = plt.subplots()
                scat = ax.scatter([], [], c='red')

                def init():
                    scat.set_offsets([])
                    return scat,

                dots_to_show = opts.sequence_length
                def animate(idx):
                    sample_idx = idx / opts.sequence_length
                    timestep_idx = idx % opts.sequence_length
                    data = samples[sample_idx, timestep_idx:timestep_idx + dots_to_show, :]
                    scat.set_offsets(data)
                    return scat,

                for idx in range(opts.num_samples):
                    ax.scatter(true_samples[idx, :, 0], true_samples[idx, :, 1], c='blue')

                animation_steps = len(samples) * opts.sequence_length
                ani = animation.FuncAnimation(fig, animate, np.arange(animation_steps), interval=10,
                            init_func=init)
                plt.show()

                for true in true_samples:
                    plt.scatter(true[:, 0], true[:, 1], c='blue')

                for gen in samples:
                    plt.scatter(gen[:, 0], gen[:, 1], c='red')

                plt.show()

                #ani.save('../media/sine.gif', writer='imagemagick', fps=30)


if __name__ == '__main__':
    unittest.main()
