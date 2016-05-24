"""
Recurrent Generative Adversarial Network implementation.

ideas:
- discriminator generates score for each timestep?
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

import learning_utils

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
        self.labels_placeholder = tf.placeholder(tf.int32, 
                                    shape=(batch_size, sequence_length), 
                                    name='labels')
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
        self.gen_scores = self.discriminate(self.generated)

        # formulate the loss
        self.gen_train_loss_out = self.gen_train_loss(self.gen_scores)

        # combine loss components
        self.total_gen_loss = self.gen_train_loss_out

        # create the gen train op
        self.gen_optimize(self.total_gen_loss)

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def build_dis_graph(self):
        # forward pass through generator to get predictions
        # returns a (batch_size, 1) output
        self.scores = self.discriminate(self.labels_placeholder, reuse=True)

        # get the loss value
        self.dis_train_loss_out = self.dis_train_loss(self.scores)

        # create the dis train op
        self.dis_optimize(self.dis_train_loss_out)

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def generate_lstm(self):
        """
        Starting with randomly sampled z, generate outputs in a 
        continuous domain.
        """
        # unpack values for easier reference
        batch_size = self.opts.batch_size
        embed_dim = self.opts.embed_dim
        vocab_dim = self.dataset.vocab_dim
        z_dim = self.opts.z_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length

        # build the generative model parameters
        with tf.variable_scope("grnn") as scope:
            # using lstm cell for rnn
            lstm = rnn.rnn_cell.BasicLSTMCell(num_hidden)

            # conversion of z into hidden state
            gWz = tf.get_variable('gWz', [z_dim, lstm.state_size])
            gbz = tf.get_variable('gbz', [lstm.state_size])

        # compute initial hidden state using sampled z value
        cell_state = tf.matmul(self.z_placeholder, gWz) + gbz

        # the label of the previously sampled word
        # shape (batch_size, 1)
        word_labels = tf.zeros((batch_size, 1), dtype=tf.int32)

        # iterate sequence_length times generating 
        # and collecting outputs in rnn_outputs
        rnn_outputs = []
        for idx in range(sequence_length):
            with tf.variable_scope("grnn") as scope:
                if idx != 0:
                    scope.reuse_variables()
                # embedding matrix
                L = tf.get_variable('L', (vocab_dim, embed_dim))

                # output to softmax weights
                gWo = tf.get_variable("gWo", (num_hidden, vocab_dim))
                gbo = tf.get_variable("gbo", (vocab_dim, ), initializer=tf.zeros)

                # retrive the word embedding and remove sequence dimension
                next_input = tf.nn.embedding_lookup(L, word_labels)
                next_input = tf.squeeze(next_input, squeeze_dims=[1])
                next_input_drop = tf.nn.dropout(next_input, self.dropout_placeholder)

                # compute the next hidden state and cell state
                hidden_state, cell_state = lstm(next_input_drop, cell_state)

                # project hidden state to size of vocabulary
                # no nonlinearity here because this will be feed into a softmax
                # as the scores for the words in the vocabulary
                scores = tf.matmul(hidden_state, gWo) + gbo

                # randomly sample from the scores
                word_labels = learning_utils.batch_sample_with_temperature(scores)
                word_labels = tf.expand_dims(word_labels, 1)

                # insert a dimension into the output to be used as concatenation dim
                rnn_outputs.append(word_labels)

        # return the word labels as shape (batch_size, sequence_length)
        rnn_outputs = tf.concat(values=rnn_outputs, concat_dim=1)
        return rnn_outputs

    def discriminate(self, labels, reuse=False):
        """
        Given a list of labels shape (batch_size, sequence_length),
        make a binary prediction as to whether each sequence in the batch is 
        real or fake.

        In order to reuse variables for both the generative and discriminative
        graphs, have reuse as argument. If True, reuses variables.
        """
        # unpack values for easier reference
        batch_size = self.opts.batch_size
        input_dim = self.opts.input_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length
        embed_dim = self.opts.embed_dim
        vocab_dim = self.dataset.vocab_dim

        # get the embeddings for all the words at once
        with tf.variable_scope("drnn", reuse=reuse) as scope:
            # embedding matrix
            L = tf.get_variable('L', (vocab_dim, embed_dim))

            # using lstm cell for rnn
            lstm = rnn.rnn_cell.BasicLSTMCell(num_hidden)

        # look up all the embeddings at once
        inputs = tf.nn.embedding_lookup(L, labels)
            
        # initialize hidden state and cell state to zeros
        hidden_state = tf.zeros((batch_size, num_hidden))
        cell_state = tf.zeros((batch_size, lstm.state_size))

        # go through inputs forward propagating them 
        # until reaching the last state, which is then 
        # used to make a decision on real vs fake
        for idx in range(sequence_length):
            with tf.variable_scope("drnn") as scope:
                # first pass through, create the lstm  
                if idx != 0 or reuse:
                    scope.reuse_variables() 

                # get the input for this timestep
                next_input = inputs[:, idx, :]
                next_input_drop = tf.nn.dropout(next_input, self.dropout_placeholder)

                # compute the next hidden state and cell state
                hidden_state, cell_state = lstm(next_input_drop, cell_state)

        # compute scores based on final hidden state
        with tf.variable_scope("drnn", reuse=reuse) as scope:            
            dWo = tf.get_variable("dWo", (num_hidden, 1))
            dbo = tf.get_variable("dbo", (1, ), initializer=tf.zeros)
            scores = tf.matmul(hidden_state, dWo) + dbo

        # for now, just return the final layer score
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
        opt = tf.train.AdamOptimizer(self.opts.learning_rate)
        self._train_gen = opt.minimize(loss)

    def dis_optimize(self, loss):
        opt = tf.train.AdamOptimizer(self.opts.learning_rate)
        self._train_dis = opt.minimize(loss)

    def get_z(self):
        # pass in z noise only once at the beginning
        z_lim = self.opts.z_lim
        z = np.random.uniform(-z_lim, z_lim, size=(self.opts.batch_size, self.opts.z_dim))
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
                feed[self.labels_placeholder] = inputs
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
        z_lim = self.opts.z_lim

        steps = 1000
        batches = steps / (batch_size *  sequence_length)
        space = np.random.uniform(-z_lim, z_lim, size=(batches, batch_size, z_dim))
        samples = []
        for batch in space:
            feed = {self.z_placeholder: batch, self.dropout_placeholder: 1}
            generated_samples = self.sess.run([self.generated], feed_dict=feed)
            samples += generated_samples[0].tolist()

        return np.array(samples)
            
    def discrete_sample_space(self):
        """
        Generate discrete samples liearly from z space.
        """
        batch_size = self.opts.batch_size
        num_samples = self.opts.num_samples
        sequence_length = self.opts.sequence_length
        z_dim = self.opts.z_dim
        z_lim = self.opts.z_lim

        steps = 1000
        batches = steps / (batch_size *  sequence_length)
        space = np.random.uniform(-z_lim, z_lim, size=(batches, batch_size, z_dim))
        samples = []
        for batch in space:
            feed = {self.z_placeholder: batch, self.dropout_placeholder: 1}
            generated_samples = self.sess.run([self.generated], feed_dict=feed)
            samples += generated_samples[0].tolist()

        return np.array(samples)




