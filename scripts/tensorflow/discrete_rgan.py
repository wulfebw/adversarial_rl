"""
DOES NOT WORK. IS NOT SUPPOSED TO WORK.

This is a test model put together to show that a GAN performing 
discrete sampling of it's output will not learn anything 
because that sampling procedure is not differentiable. 
This problem is solved by training the GAN with policy gradient 
methods as shown in a different file.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

import learning_utils

class RecurrentDiscreteGenerativeAdversarialNetwork(object):
    """
    GAN implementation for DISCRETE output. This model does 
    and should not work because the sampling procedure is 
    not differentiable. 
    """

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
        self.temperature_placeholder = tf.placeholder(tf.float32, shape=(), name='temperature')

    def build_gen_graph(self):
        # forward pass through generator
        # returns a (batch_size, sequence_length, input_dim) for generated
        self.generated, self.timestep_probs = self.generate_lstm()

        # get the predictions from the discriminator
        # returns a (batch_size, 1) output
        self.gen_scores = self.discriminate(self.generated)

        # formulate the loss
        self.gen_train_loss_out = self.gen_train_loss(self.gen_scores)

        # create the gen train op
        self.gen_optimize(self.gen_train_loss_out)

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
        timestep_probs = []
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

                # randomly sample from the scores and also get the probabilities
                word_labels, probs = learning_utils.batch_sample_with_temperature(scores, 
                                        self.temperature_placeholder)

                # insert a dimension along which to accumlate
                word_labels = tf.expand_dims(word_labels, 1)
                probs = tf.expand_dims(probs, 1)

                # accumulate the output word labels as well as 
                # the probabilities that generated those labels
                rnn_outputs.append(word_labels)
                timestep_probs.append(probs)

        # return the word labels as shape (batch_size, sequence_length)
        # return the probs at each timestep as shape 
        # (batch_size, sequence_length, vocab_dim)
        rnn_outputs = tf.concat(values=rnn_outputs, concat_dim=1)
        timestep_probs = tf.concat(values=timestep_probs, concat_dim=1)
        return rnn_outputs, timestep_probs

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
        # loss = tf.reduce_mean(-tf.log(probs))
        loss = -tf.log(probs)
        loss = tf.squeeze(loss, squeeze_dims=[1])
        return loss

    def dis_train_loss(self, scores):
        probs = tf.nn.sigmoid(scores)
        # minimize binary cross entropy 
        ce = -(self.targets_placeholder * tf.log(probs) + 
            (1 - self.targets_placeholder) * tf.log(1 - probs))
        loss = tf.reduce_mean(ce)
        return loss

    def gen_optimize(self, loss):

        # for now, only propagate loss for "action" aka word selection
        # at last timestep. Here, get that index and the resultant gradient shape
        final_timestep = self.opts.sequence_length - 1
        final_probs = self.timestep_probs[:, final_timestep, :]
        grad_shape = final_probs.get_shape()

        # create an array of indices 
        # where for each row, the first column is just the row number 
        # and the second column is the index of the selected word during sampling
        # basically, this is the numpy arange indexing with a matrix trick
        word_idxs = tf.expand_dims(self.generated[:, final_timestep], 1)
        range_idxs = tf.to_int64(tf.expand_dims(tf.range(self.opts.batch_size), 1))
        indices = tf.concat(1, (range_idxs, word_idxs))

        # create the actual gradients
        # do so by creating a vector of the rewards aka losses 
        # inserted at each row in the column of the sampled word
        # then average over the rows to get the gradients across 
        # the entire minibatch 
        grads = tf.SparseTensor(indices, loss, grad_shape)
        grads = tf.sparse_tensor_to_dense(grads)
        grads = tf.mul(grads, final_probs)

        # option 1: get and apply the gradients manually
        # grads = tf.reduce_mean(grads, 0)
        # params = tf.trainable_variables()
        # params = [p for p in params if 'grnn' in p.name]
        # gradients = tf.gradients(grads, params)
        # print gradients
        # raw_input()

        # option 2: treat the mean of grads as the loss and minimize it
        pgloss = tf.reduce_mean(grads)
        opt = tf.train.AdamOptimizer(self.opts.learning_rate)        
        self._train_gen = opt.minimize(pgloss)

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
            feed[self.temperature_placeholder] = self.opts.temperature

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
                feed[self.dropout_placeholder] = self.opts.dropout

                # perform the actual training step if training
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
            feed = {self.z_placeholder: batch, 
                    self.dropout_placeholder: 1}
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

        steps = 10000
        batches = steps / (batch_size *  sequence_length)
        space = np.random.uniform(-z_lim, z_lim, size=(batches, batch_size, z_dim))
        samples = []
        for batch in space:
            feed = {self.z_placeholder: batch, 
                    self.dropout_placeholder: 1,
                    self.temperature_placeholder: .6}
            output_values = [self.generated, self.timestep_probs]
            generated_samples, timestep_probs = self.sess.run(output_values, feed_dict=feed)
            samples += generated_samples.tolist()

        return np.array(samples)




