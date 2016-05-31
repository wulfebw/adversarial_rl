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
        self.max_norm = 2.0
        self.decay_every = 10000
        self.decay_ratio = .96

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
        opts.learning_rate = .01
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 5
        opts.batch_size = 52
        opts.num_samples = 130
        opts.epochs_to_train = 1000
        opts.num_hidden = 128
        opts.embed_dim = 32
        opts.z_dim = 2
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .5
        opts.full_sequence_optimization = True
        opts.save_every = 200
        opts.plot_every = 50
        opts.reduce_temperature_every = 20
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 25
        opts.decay_ratio = .96
        opts.max_norm = 5.0

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            # get the param values beforehand
            params = tf.trainable_variables()
            # param_info = sorted([(p.name, p.eval()) for p in params 
            #                     if 'grnn' in p.name])
            param_info = sorted([(p.name, p.eval()) for p in params])

            # train
            losses = []
            TRAIN = False
            if TRAIN == True:
                for epoch in range(opts.epochs_to_train):
                    model.run_epoch()  

                    if epoch % opts.save_every == 0:
                        saver.save(session, '../snapshots/{}.weights'.format(opts.dataset_name))

                    if epoch % opts.plot_every == 0:
                        model.plot_results()

                    if epoch % opts.reduce_temperature_every == 0:
                        opts.temperature -= opts.temperature_reduction_amount
                        opts.temperature = max(opts.min_temperature, opts.temperature)

            # sample linearly from z space
            if opts.discrete:
                samples, probs = model.discrete_sample_space()
                samples = dataset.decode_dataset(samples)
            else:
                samples = model.sample_space()

            # get the samples in the dataset
            true_samples = dataset.data['X_train']
            true_samples = dataset.decode_dataset(true_samples)

            # calculate and report metric
            total = float(len(samples))
            in_real_data_count = 0
            less_than_count = 0
            for sample in samples:
                if sample in true_samples:
                    in_real_data_count += 1
                if sample[0] < sample[1]:
                    less_than_count += 1

            num_display = 10
            for (s, p) in zip(samples[:num_display], probs[:num_display]):
                print "example generated data: {}".format(s)
                print "probabilities of those selections: {}".format(p)
            print "total samples: {}".format(total)
            print "generated samples also in dataset: {}".format(in_real_data_count)
            print "percent generated in real dataset: {}%".format(100 * in_real_data_count / total)
            print "percent samples[0] < samples[1]: {}%".format(100 * less_than_count / total)

            if SHOW_PLOTS:
                model.plot_results()

            # assert that the parameters of the generative model have not changed
            # at all because they cannot possibly change because they are blocked
            # from any gradient by a nondifferentiable, discrete sampling operation
            params_after = tf.trainable_variables()
            # param_after_info = sorted([(p.name, p.eval()) for p in params_after 
            #                             if 'grnn' in p.name])
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
                # print vals.flatten().tolist()
                # print vals_after.flatten().tolist()
                num_params = len(vals_after.flatten().tolist())
                total_num_params += num_params
                diffs = vals - vals_after
                diff = np.sum(np.abs(diffs))
                total_diff += diff
                print "average absolute difference: {}%".format(diff / num_params * 100)

            print "overall average absolute difference: {}%".format(total_diff / total_num_params * 100)

    def test_run_epoch_mlb(self):
        """
        Test with both training.
        """
        opts = TestOptions()    
        opts.learning_rate = .01
        opts.epoch_multiple_gen = 2
        opts.epoch_multiple_dis = 1
        opts.batch_size = 32
        opts.sequence_length = 3
        opts.num_samples = 512
        opts.epochs_to_train = 100000
        opts.num_hidden = 256
        opts.embed_dim = 32
        opts.z_dim = 4
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .1
        opts.full_sequence_optimization = True
        opts.save_every = 200
        opts.plot_every = 50
        opts.reduce_temperature_every = 100
        opts.temperature_reduction_amount = .001
        opts.min_temperature = .1
        opts.decay_every = 50
        opts.decay_ratio = .99
        opts.max_norm = 2.0
        opts.sentence_limit = 1
        opts.dataset_name = 'mlb'

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            # get the param values beforehand
            params = tf.trainable_variables()
            # param_info = sorted([(p.name, p.eval()) for p in params 
            #                     if 'grnn' in p.name])
            param_info = sorted([(p.name, p.eval()) for p in params])

            # train
            losses = []
            TRAIN = False
            if TRAIN == True:
                for epoch in range(opts.epochs_to_train):
                    model.run_epoch()  

                    if epoch % opts.save_every == 0:
                        saver.save(session, '../snapshots/{}.weights'.format(opts.dataset_name))

                    if epoch % opts.plot_every == 0:
                        model.plot_results()

                    if epoch % opts.reduce_temperature_every == 0:
                        opts.temperature -= opts.temperature_reduction_amount
                        opts.temperature = max(opts.min_temperature, opts.temperature)

            # sample linearly from z space
            if opts.discrete:
                samples, probs = model.discrete_sample_space()
                samples = dataset.decode_dataset(samples, real=True)
            else:
                samples = model.sample_space()

            # get the samples in the dataset
            true_samples = dataset.data['X_train']
            true_samples = dataset.decode_dataset(true_samples, real=True)

            # calculate and report metric
            total = float(len(samples))
            in_real_data_count = 0
            less_than_count = 0
            for sample in samples:
                if sample in true_samples:
                    in_real_data_count += 1
                if sample[0] < sample[1]:
                    less_than_count += 1

            num_display = 10
            for (s, p) in zip(samples[:num_display], probs[:num_display]):
                print "example generated data: {}".format(s)
                print "probabilities of those selections: {}".format(p)
            print "total samples: {}".format(total)
            print "generated samples also in dataset: {}".format(in_real_data_count)
            print "percent generated in real dataset: {}%".format(100 * in_real_data_count / total)
            print "percent samples[0] < samples[1]: {}%".format(100 * less_than_count / total)

            if SHOW_PLOTS:
                model.plot_results()

            # assert that the parameters of the generative model have not changed
            # at all because they cannot possibly change because they are blocked
            # from any gradient by a nondifferentiable, discrete sampling operation
            params_after = tf.trainable_variables()
            # param_after_info = sorted([(p.name, p.eval()) for p in params_after 
            #                             if 'grnn' in p.name])
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
                # print vals.flatten().tolist()
                # print vals_after.flatten().tolist()
                num_params = len(vals_after.flatten().tolist())
                total_num_params += num_params
                diffs = vals - vals_after
                diff = np.sum(np.abs(diffs))
                total_diff += diff
                print "average absolute difference: {}%".format(diff / num_params * 100)

            print "overall average absolute difference: {}%".format(total_diff / total_num_params * 100)

if __name__ == '__main__':
    unittest.main()
