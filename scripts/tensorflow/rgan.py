"""
Adversarial networks implementation
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

Z_LIM = np.pi * 2

class RecurrentGenerativeAdversarialNetwork(object):

    def __init__(self, options, session, dataset):
        self.opts = options
        self.sess = session
        self.dataset = dataset

        self.build_placeholders()
        self.build_gen_graph()
        self.build_dis_graph()

        self.epoch = 0
        self.updates = 0
        self.train_gen_losses = []
        self.train_dis_losses = []

    def build_placeholders(self):
        # unpack values for easier reference
        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        input_dim = self.opts.input_dim
        z_dim = self.opts.z_dim

        # placeholder for z, randomly sampled data
        self.z_placeholder = tf.placeholder(tf.float32, shape=(batch_size, z_dim), name='z')
        # placeholder for inputs
        self.inputs_placeholder = tf.placeholder(tf.float32, 
                                    shape=(batch_size, sequence_length, input_dim), 
                                    name='inputs')
        # placeholder for target values
        self.targets_placeholder = tf.placeholder(tf.float32, 
                                    shape=(batch_size, 1), 
                                    name='targets')
        # placeholder for dropout keep fraction
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout')

    def build_gen_graph(self):
        # forward pass through generator
        # returns a (batch_size, sequence_length, input_dim) for generated
        self.generated = self.generate_lstm()

        # get the predictions from the discriminator
        # returns a (batch_size, 1) output
        self.gen_predictions = self.discriminate(self.generated)

        # formulate the loss
        self.gen_train_loss_out = self.gen_train_loss(self.gen_predictions)

        # create the gen train op
        self.gen_optimize(self.gen_train_loss_out)

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def build_dis_graph(self):
        # forward pass through generator to get predictions
        # returns a (batch_size, 1) output
        self.predictions = self.discriminate(self.inputs_placeholder, reuse=True)

        # get the loss value
        self.dis_train_loss_out = self.dis_train_loss(self.predictions)

        # create the dis train op
        self.dis_optimize(self.dis_train_loss_out)

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def generate(self):
        """
        Starting with randomly sampled z, generate outputs in a 
        continuous domain.
        """
        # unpack values for easier reference
        batch_size = self.opts.batch_size
        input_dim = self.opts.input_dim
        z_dim = self.opts.z_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length

        # compute initial hidden state using sampled z value
        gWz = tf.get_variable('gWz', [z_dim, num_hidden])
        gbz = tf.get_variable('gbz', [num_hidden])
        prev_h = tf.nn.tanh(tf.matmul(self.z_placeholder, gWz) + gbz)

        # set initial input value to be zeros
        # possible that it would be better to use z here as well
        prev_output = tf.zeros((batch_size, input_dim))

        # iterate sequence_length time generating and collecting
        # outputs in rnn_outputs
        rnn_outputs = []
        for idx in range(sequence_length):
            with tf.variable_scope("grnn") as scope:
                if idx != 0:
                    scope.reuse_variables()

                gWi = tf.get_variable("gWi", (input_dim, num_hidden))
                gWh = tf.get_variable("gWh", (num_hidden, num_hidden))
                gbh = tf.get_variable("gbh", (num_hidden, ), initializer=tf.zeros)
                gWo = tf.get_variable("gWo", (num_hidden, input_dim))
                gbo = tf.get_variable("gbo", (input_dim, ), initializer=tf.zeros)

            prev_h = tf.nn.tanh(tf.matmul(prev_output, gWi) + tf.matmul(prev_h, gWh) + gbh)
            prev_output = tf.matmul(prev_h, gWo) + gbo

            # insert a dimension into the output to be used as concatenation dim
            rnn_outputs.append(tf.expand_dims(prev_output, 1))

        rnn_outputs = tf.concat(values=rnn_outputs, concat_dim=1)
        return rnn_outputs

    def generate_lstm(self):
        """
        Starting with randomly sampled z, generate outputs in a 
        continuous domain.
        """
        # unpack values for easier reference
        batch_size = self.opts.batch_size
        input_dim = self.opts.input_dim
        z_dim = self.opts.z_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length

        # use lstm cell
        lstm = rnn.rnn_cell.BasicLSTMCell(num_hidden)

        # compute initial hidden state using sampled z value
        gWz = tf.get_variable('gWz', [z_dim, lstm.state_size])
        gbz = tf.get_variable('gbz', [lstm.state_size])
        cell_state = tf.nn.tanh(tf.matmul(self.z_placeholder, gWz) + gbz)

        # set initial input value to be zeros
        # possible that it would be better to use z here as well
        prev_output = tf.zeros((batch_size, input_dim))

        # iterate sequence_length time generating and collecting
        # outputs in rnn_outputs
        rnn_outputs = []
        for idx in range(sequence_length):
            with tf.variable_scope("grnn") as scope:
                if idx != 0:
                    scope.reuse_variables()

                gWi = tf.get_variable("gWi", (input_dim, num_hidden))
                gbi = tf.get_variable("gbi", (num_hidden, ), initializer=tf.zeros)
                gWo = tf.get_variable("gWo", (num_hidden, input_dim))
                gbo = tf.get_variable("gbo", (input_dim, ), initializer=tf.zeros)

                next_input = tf.nn.tanh(tf.matmul(prev_output, gWi) + gbi)
                hidden_state, cell_state = lstm(next_input, cell_state)
                prev_output = tf.matmul(hidden_state, gWo) + gbo

            # insert a dimension into the output to be used as concatenation dim
            rnn_outputs.append(tf.expand_dims(prev_output, 1))

        rnn_outputs = tf.concat(values=rnn_outputs, concat_dim=1)
        return rnn_outputs

    def discriminate(self, inputs, reuse=False):
        """
        Given a set of inputs shape (batch_size, sequence_length, input_dim),
        make a binary prediction as to whether it is real or fake data.

        In order to reuse variables for both the generative and discriminative
        graphs, have reuse as argument. If True, reuses variables
        """

        # unpack values for easier reference
        batch_size = self.opts.batch_size
        input_dim = self.opts.input_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length

        # initialize hidden state to zeros
        prev_h = tf.zeros((batch_size, num_hidden))

        # go through inputs forward propagating them 
        # until reaching the last state, which is then 
        # used to make a decision on real vs fake
        for idx in range(sequence_length):
            with tf.variable_scope("drnn") as scope:
                if idx != 0 or reuse:
                    scope.reuse_variables()

                dWi = tf.get_variable("dWi", (input_dim, num_hidden))
                dWh = tf.get_variable("dWh", (num_hidden, num_hidden))
                dbh = tf.get_variable("dbh", (num_hidden, ), initializer=tf.zeros)

            x = inputs[:, idx, :]
            prev_h = tf.nn.tanh(tf.matmul(x, dWi) + tf.matmul(prev_h, dWh) + dbh)

        # get the output variables, reusing if this is for the discriminator's graph
        with tf.variable_scope("drnn") as scope:
            if reuse:
                scope.reuse_variables()
            dWo = tf.get_variable("dWo", (num_hidden, 1))
            dbo = tf.get_variable("dbo", (1, ), initializer=tf.zeros)
        scores = tf.matmul(prev_h, dWo) + dbo

        return scores

    def gen_train_loss(self, scores):
        probs = tf.nn.sigmoid(scores)
        # want to _maximize_ the discriminator's probability outputs
        # so _minimize_ the negative of the log of the outputs
        loss = tf.reduce_mean(-tf.log(probs))
        return loss

    def dis_train_loss(self, scores):
        probs = tf.nn.sigmoid(scores)
        # minimize binary cross entropy 
        ce = -(self.targets_placeholder * tf.log(probs) + 
            (1 - self.targets_placeholder) * tf.log(1 - probs))
        loss = tf.reduce_mean(ce)
        return loss

    def gen_optimize(self, loss):
        # define optimizer
        opt = tf.train.AdamOptimizer(self.opts.learning_rate)
        self._train_gen = opt.minimize(loss)

    def dis_optimize(self, loss):
        # define optimizer
        opt = tf.train.AdamOptimizer(self.opts.learning_rate)
        self._train_dis = opt.minimize(loss)

    def get_z(self):
        # pass in z noise only once at the beginning
        z = np.random.uniform(-Z_LIM, Z_LIM, size=(self.opts.batch_size, self.opts.z_dim))
        return z

    def train_generator(self):
        losses = []
        num_batches = self.opts.num_samples / self.opts.batch_size
        for bidx in range(num_batches):
            
            # build the dict to feed inputs to graph
            feed = {}
            z = self.get_z()
            feed[self.z_placeholder] = z
            feed[self.dropout_placeholder] = self.opts.dropout

            # perform the actual training step if training
            output_values = [self._train_gen, self.gen_train_loss_out, self.generated]
            _, loss_out, generated = self.sess.run(output_values, feed_dict=feed)

            # add generated samples to dataset for training discriminator
            self.dataset.add_generated_samples(generated)
            losses.append(loss_out)

        return losses

    def train_discriminator(self):
        losses = []

        for depoch in range(self.opts.train_ratio):
            for inputs, targets, in self.dataset.next_batch():

                # build the dict to feed inputs to graph
                feed = {}
                feed[self.inputs_placeholder] = inputs
                feed[self.targets_placeholder] = targets

                # perform the actual training step if training
                feed[self.dropout_placeholder] = self.opts.dropout
                output_values = [self._train_dis, self.dis_train_loss_out]
                _, loss_out = self.sess.run(output_values, feed_dict=feed)

                losses.append(loss_out)

        self.dataset.reset_generated_samples()
        self.epoch += 1
        return losses
        
    def run_epoch(self):
        # for now, track losses manually
        gen_losses = []
        dis_losses = []

        gen_losses_out = self.train_generator()
        dis_losses_out = self.train_discriminator()

        gen_losses.append(gen_losses_out)
        dis_losses.append(dis_losses_out)

        mean_gen_loss = np.mean(gen_losses_out)
        mean_dis_loss = np.mean(dis_losses_out)

        print('train epoch: {}\tgen loss: {}\tdis loss: {}'.format(
            self.epoch, mean_gen_loss, mean_dis_loss))

        self.train_gen_losses.append(mean_gen_loss)
        self.train_dis_losses.append(mean_dis_loss)

    def plot_results(self):
        plt.plot(np.array(self.train_gen_losses), c='blue', linestyle='solid',
            label='training gen loss')
        plt.plot(np.array(self.train_dis_losses), c='red', linestyle='solid', 
            label='training dis loss')
        plt.legend()
        plt.show()

    def sample_space(self):
        """
        Generates samples linearly from z space.
        """
        batch_size = self.opts.batch_size
        num_samples = self.opts.num_samples
        sequence_length = self.opts.sequence_length
        z_dim = self.opts.z_dim

        steps = 1000
        batches = steps / (batch_size *  sequence_length)
        space = np.random.uniform(-Z_LIM, Z_LIM, size=(batches, batch_size, z_dim))
        samples = []
        for batch in space:
            feed = {self.z_placeholder: batch, self.dropout_placeholder: 1}
            generated_samples = self.sess.run([self.generated], feed_dict=feed)
            samples += generated_samples[0].tolist()

        return np.array(samples)
            




