
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/tensorflow')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts/utils')))

import aws_s3_utility
import datasets
import discrete_rgan
import learning_utils

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
        self.num_samples = 100
        self.max_norm = 2.0
        self.decay_every = 10000
        self.decay_ratio = .96
        self.epoch_multiple_gen = 1
        self.epoch_multiple_dis = 1

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
        opts.z_dim = 16
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .2
        opts.full_sequence_optimization = True
        opts.save_every = 200
        opts.plot_every = 100
        opts.reduce_temperature_every = 20
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 10
        opts.decay_ratio = .96
        opts.max_norm = 1.0
        opts.pretrain_learning_rate = .01

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            # get the param values beforehand
            params = tf.trainable_variables()
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
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
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
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 2
        opts.batch_size = 16
        opts.sequence_length = 4
        opts.num_samples = 500
        opts.epochs_to_train = 1000
        opts.num_hidden = 256
        opts.embed_dim = 50
        opts.z_dim = 20
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .1
        opts.full_sequence_optimization = True
        opts.save_every = 200
        opts.plot_every = 50
        opts.reduce_temperature_every = 100
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 100
        opts.decay_ratio = .96
        opts.max_norm = 2.0
        opts.pretrain_learning_rate = .001
        opts.sentence_limit = 1
        opts.dataset_name = 'mlb'

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            saver.restore(session, '../snapshots/{}.weights'.format(opts.dataset_name))

            # get the param values beforehand
            params = tf.trainable_variables()
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

            perplexity = learning_utils.calculate_perplexity(probs)

            num_display = 20
            for (s, p) in zip(samples[:num_display], probs[:num_display]):
                print "example generated data: {}".format(s)
                # print "probabilities of those selections: {}".format(p)
            print "total samples: {}".format(total)
            print "generated samples also in dataset: {}".format(in_real_data_count)
            print "percent generated in real dataset: {}%".format(100 * in_real_data_count / total)
            print "perplexity of samples: {}".format(perplexity)

            if SHOW_PLOTS:
                model.plot_results()

            # assert that the parameters of the generative model have not changed
            # at all because they cannot possibly change because they are blocked
            # from any gradient by a nondifferentiable, discrete sampling operation
            params_after = tf.trainable_variables()
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
                num_params = len(vals_after.flatten().tolist())
                total_num_params += num_params
                diffs = vals - vals_after
                diff = np.sum(np.abs(diffs))
                total_diff += diff
                print "average absolute difference: {}%".format(diff / num_params * 100)

            print "overall average absolute difference: {}%".format(total_diff / total_num_params * 100)

    def test_run_epoch_mlb_with_pretraining(self):
        """
        Test with both training.
        """
        opts = TestOptions()    
        opts.learning_rate = .001
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 2
        opts.batch_size = 32
        opts.sequence_length = 4
        opts.num_samples = 64
        opts.epochs_to_train = 10
        opts.num_hidden = 16
        opts.embed_dim = 8
        opts.z_dim = 20
        opts.dropout = 1.
        opts.temperature = .2
        opts.sampling_temperature = .2
        opts.full_sequence_optimization = True
        opts.save_every = 10
        opts.plot_every = 50
        opts.reduce_temperature_every = 100
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 100
        opts.decay_ratio = .96
        opts.max_norm = 2.0
        opts.sentence_limit = 1
        opts.pretrain_epochs = 50
        opts.pretrain_learning_rate = .001
        opts.dataset_name = 'mlb'

        ak = aws_s3_utility.load_key('../keys/access_key.key')
        sk = aws_s3_utility.load_key('../keys/secret_key.key')
        bucket = 'pgrgan'
        aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
        weights_filepath = '../snapshots/{}.weights'.format(opts.dataset_name)
        weights_filename = '{}.weights'.format(opts.dataset_name)

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            #saver.restore(session, weights_filepath)

            # get the param values beforehand
            params = tf.trainable_variables()
            param_info = sorted([(p.name, p.eval()) for p in params])

            # train
            losses = []
            TRAIN = True
            if TRAIN == True:
                # for epoch in range(opts.pretrain_epochs):
                #     model.run_pretrain_epoch()

                for epoch in range(opts.epochs_to_train):
                    model.run_pretrain_epoch()
                    model.run_epoch()  

                    if epoch % opts.save_every == 0:
                        saver.save(session, weights_filepath)
                        aws_util.upload_file(weights_filename, weights_filepath)

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

            perplexity = learning_utils.calculate_perplexity(probs)

            num_display = 3
            for (s, p) in zip(samples[:num_display], probs[:num_display]):
                print "example generated data: {}".format(s)
                print "probabilities of those selections: {}".format(p)
            print "total samples: {}".format(total)
            print "generated samples also in dataset: {}".format(in_real_data_count)
            print "percent generated in real dataset: {}%".format(100 * in_real_data_count / total)
            print "perplexity of samples: {}".format(perplexity)

            if SHOW_PLOTS:
                model.plot_results()

            # assert that the parameters of the generative model have not changed
            # at all because they cannot possibly change because they are blocked
            # from any gradient by a nondifferentiable, discrete sampling operation
            params_after = tf.trainable_variables()
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
                num_params = len(vals_after.flatten().tolist())
                total_num_params += num_params
                diffs = vals - vals_after
                diff = np.sum(np.abs(diffs))
                total_diff += diff
                print "average absolute difference: {}%".format(diff / num_params * 100)

            print "overall average absolute difference: {}%".format(total_diff / total_num_params * 100)
    def test_run_epoch_twitch_with_pretraining(self):
        """
        Test with both training.
        """
        opts = TestOptions()    
        opts.learning_rate = .005
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 1
        opts.batch_size = 64
        opts.sequence_length = 4
        opts.num_samples = 100000
        opts.epochs_to_train = 1000
        opts.num_hidden = 256
        opts.embed_dim = 64
        opts.z_dim = 16
        opts.dropout = .9
        opts.temperature = 1.
        opts.sampling_temperature = .5
        opts.full_sequence_optimization = True
        opts.save_every = 1
        opts.plot_every = 1
        opts.reduce_temperature_every = 1
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 100
        opts.decay_ratio = .97
        opts.max_norm = 2.0
        opts.sentence_limit = 50000
        opts.pretrain_epochs = 50
        opts.pretrain_learning_rate = .005
        opts.save_to_aws = False
        opts.dataset_name = 'twitch'
        opts.aws_bucket = 'pgrganxentbaseline'
        opts.with_baseline = True
        opts.with_xent = True
        opts.sample_every = 2

        ak = aws_s3_utility.load_key('../keys/access_key.key')
        sk = aws_s3_utility.load_key('../keys/secret_key.key')
        bucket = opts.aws_bucket
        aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
        weights_filepath = '../snapshots/{}.weights'.format(opts.dataset_name)
        weights_filename = '{}.weights'.format(opts.dataset_name)

        with tf.Session() as session:
            dataset = datasets.FakeRecurrentAdversarialDataset(opts)
            model = discrete_rgan.RecurrentDiscreteGenerativeAdversarialNetwork(opts, session, dataset)
            saver = tf.train.Saver()
            #saver.restore(session, weights_filepath)

            # get the param values beforehand
            params = tf.trainable_variables()
            param_info = sorted([(p.name, p.eval()) for p in params])

            # train
            losses = []
            TRAIN = True
            if TRAIN == True:
                for epoch in range(opts.epochs_to_train):

                    if opts.with_xent:
                        model.run_pretrain_epoch()
                    model.run_epoch()  

                    if epoch % opts.save_every == 0:
                        saver.save(session, weights_filepath)
                        if opts.save_to_aws:
                            try:
                                aws_util.upload_file(weights_filename, weights_filepath)
                            except:
                                pass

                    if epoch % opts.plot_every == 0:
                        model.plot_results()
                        if opts.save_to_aws:
                            try:
                                aws_util.upload_directory('../media')
                            except:
                                pass

                    if epoch % opts.reduce_temperature_every == 0:
                        opts.temperature -= opts.temperature_reduction_amount
                        opts.temperature = max(opts.min_temperature, opts.temperature)

                    if epoch % opts.sample_every == 0:
                        samples, probs = model.discrete_sample_space()
                        samples = dataset.decode_dataset(samples, real=True)
                        print samples[0]
                        np.savez('../media/samples.npz', samples=samples, probs=probs)

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

            perplexity = 0 #learning_utils.calculate_perplexity(probs)

            num_display = 30
            for (s, p) in zip(samples[:num_display], probs[:num_display]):
                print "example generated data: {}".format(s)
                #print "probabilities of those selections: {}".format(p)
            print "total samples: {}".format(total)
            print "generated samples also in dataset: {}".format(in_real_data_count)
            print "percent generated in real dataset: {}%".format(100 * in_real_data_count / total)
            print "perplexity of samples: {}".format(perplexity)

            if SHOW_PLOTS:
                model.plot_results()

            # assert that the parameters of the generative model have not changed
            # at all because they cannot possibly change because they are blocked
            # from any gradient by a nondifferentiable, discrete sampling operation
            params_after = tf.trainable_variables()
            param_after_info = sorted([(p.name, p.eval()) for p in params_after])
            total_diff = 0
            total_num_params = 0
            for (n, vals), (n_after, vals_after) in zip(param_info, param_after_info):
                print "\n"
                print n
                print n_after
                num_params = len(vals_after.flatten().tolist())
                total_num_params += num_params
                diffs = vals - vals_after
                diff = np.sum(np.abs(diffs))
                total_diff += diff
                print "average absolute difference: {}%".format(diff / num_params * 100)

            print "overall average absolute difference: {}%".format(total_diff / total_num_params * 100)

if __name__ == '__main__':
    unittest.main()


