"""
confirming it works:
1. go through math and justify
2. visualize stuff
    - hidden state
    - impact of z on outputs
    - discriminators opinion of different samples
    - figure out what the loss values should be and what they should tend towards
    - plot of percent of generated samples in true dataset as function of 
        - number of actions
        - temperature

ideas:
1. use a baseline to reduce variance of sample reward [next]
2. convert to actor critic by treating the hidden state as the state
3. backprop reward at each timestep [check]
4. reduce temperature over time [check]
5. make separate discriminator? [check]
6. gradient clipping [check]
7. why do values go to nan? [check, b/c numerical statbility in both loss and multinomial]
8. figure out if multinomial matches [check]
9. simultaneous training with ce loss
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq
from tensorflow.python.ops.seq2seq import sequence_loss

import learning_utils

class RecurrentDiscreteGenerativeAdversarialNetwork(object):
    """
     
    """

    def __init__(self, options, session, dataset):
        self.opts = options
        self.sess = session
        self.dataset = dataset

        self.build_placeholders()
        self.build_gen_graph()
        self.build_dis_graph()
        self.build_xent_graph()

        self.epoch = 0
        self.updates = 0
        self.train_gen_losses = []
        self.train_dis_losses = []
        self.perplexities = []
        self.baseline_losses = []
        self.pretrain_losses = []

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
        self.generated, self.timestep_probs, self.predicted_rewards = self.generate()

        # get the predictions from the discriminator
        # returns a (batch_size, 1) output
        self.gen_scores = self.discriminate(self.generated, reuse=False)

        # formulate the policy gradient loss
        self.gen_train_loss_out, self.baseline_loss = self.gen_train_loss(self.gen_scores,
             self.predicted_rewards)

        # get generative parameters and baseline params
        self.g_params = [p for p in tf.trainable_variables() if 'g' in p.name and 'b' not in p.name]
        self.b_params = [p for p in tf.trainable_variables() if 'b' in p.name]

        # create the gen train op
        self.gen_optimize_rewards(self.gen_train_loss_out)

        # create the baseline train op
        self.optimize_baseline(self.baseline_loss)

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def build_dis_graph(self):
        # forward pass through generator to get predictions
        # returns a (batch_size, 1) output
        self.scores = self.discriminate(self.labels_placeholder, reuse=True)

        # get the loss value
        self.dis_train_loss_out = self.dis_train_loss(self.scores)

        # get discriminative parameters
        self.d_params = [p for p in tf.trainable_variables() if 'd' in p.name]

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
        timestep_predicted_rewards = []
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

                # predict the rewards resultant from this hidden state
                # block gradient from propagating into hidden state
                gWb = tf.get_variable("gWb", (num_hidden, 1))
                gbb = tf.get_variable("gbb", (1, ), initializer=tf.zeros)
                predicted_rewards = tf.matmul(tf.stop_gradient(hidden_state), gWb) + gbb

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
                timestep_predicted_rewards.append(predicted_rewards)

        # return the word labels as shape (batch_size, sequence_length)
        # return the probs at each timestep as shape 
        # (batch_size, sequence_length, vocab_dim)
        rnn_outputs = tf.concat(values=rnn_outputs, concat_dim=1)
        timestep_probs = tf.concat(values=timestep_probs, concat_dim=1)
        timestep_predicted_rewards = tf.concat(values=timestep_predicted_rewards, concat_dim=1)
        return rnn_outputs, timestep_probs, timestep_predicted_rewards

    def discriminate(self, labels, reuse):
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
        with tf.variable_scope('drnn', reuse=reuse) as scope:
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
        rnn_scores = []
        for idx in range(sequence_length):
            with tf.variable_scope('drnn', reuse=reuse) as scope:
                # first pass through, create the lstm  
                if idx != 0 or reuse:
                    scope.reuse_variables() 

                # create output params
                dWo = tf.get_variable("dWo", (num_hidden, 1))
                dbo = tf.get_variable("dbo", (1, ), initializer=tf.zeros)

                # get the input for this timestep
                next_input = inputs[:, idx, :]
                next_input_drop = tf.nn.dropout(next_input, self.dropout_placeholder)

                # compute the next hidden state and cell state
                hidden_state, cell_state = lstm(next_input_drop, cell_state)

                # compute output score for this timestep
                scores = tf.matmul(hidden_state, dWo) + dbo
                rnn_scores.append(tf.expand_dims(scores, 1))

        # for now, just return the final layer score
        rnn_scores = tf.concat(values=rnn_scores, concat_dim=1)
        return rnn_scores

    def gen_train_loss(self, scores, predicted_rewards):
        # want to _maximize_ the discriminator's probability outputs
        # so _minimize_ the negative of the log of the outputs
        probs = tf.nn.sigmoid(scores)
        rewards = -tf.log(probs)
        rewards = tf.squeeze(rewards, squeeze_dims=[2])

        # subtract baseline
        baseline_subtracted_rewards = rewards - predicted_rewards
        baseline_loss = tf.reduce_mean(baseline_subtracted_rewards ** 2)

        # policy gradient loss is, for each reward, that reward
        # times the probability of the action that resulted in that reward

        # not sure how to incorporate a baseline, might be able to just do it here
        # by subtracting from the reward value
        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        vocab_dim = self.dataset.vocab_dim

        total_loss = tf.constant(0.)
        for timestep in range(sequence_length):
            # get the probabilities for this timestep
            probs = self.timestep_probs[:, timestep, :]

            # create indices into sparse tensor for this timestep
            # where for each row, the first column is just the row number 
            # and the second column is the index of the selected word during sampling
            # basically, this is the numpy arange indexing with a matrix trick
            word_idxs = tf.expand_dims(self.generated[:, timestep], 1)
            range_idxs = tf.to_int64(tf.expand_dims(tf.range(batch_size), 1))
            indices = tf.concat(1, (range_idxs, word_idxs))
            choosen_word_indicators = tf.ones((batch_size,))

            choosen_word_probs = tf.SparseTensor(indices, choosen_word_indicators, probs.get_shape())
            choosen_word_probs = tf.sparse_tensor_to_dense(choosen_word_probs)
            choosen_word_probs = tf.reduce_sum(tf.mul(choosen_word_probs, probs), 
                                    reduction_indices=1)

            # get the rewards for this timestep
            #timestep_rewards = rewards[:, timestep]
            timestep_rewards = baseline_subtracted_rewards[:, timestep]
            
            # compute loss this timestep
            timestep_loss = tf.mul(timestep_rewards, choosen_word_probs)
            total_loss += timestep_loss

        return total_loss, baseline_loss

    def dis_train_loss(self, scores):
        # sigmoid scores to get positive class probabilities
        probs = tf.nn.sigmoid(scores)
        probs = tf.squeeze(probs, squeeze_dims=[2])

        self.dis_probs = probs

        # minimize binary cross entropy 
        # add 1e-6 to second log to prevent it from going to infinity
        self.ce = -(self.targets_placeholder * tf.log(probs) + 
            (1 - self.targets_placeholder) * tf.log(1 - probs + 1e-6))
        loss = tf.reduce_mean(self.ce)
        return loss

    def gen_optimize_rewards(self, loss):

        # minimize the loss over the full sequence
        global_step = tf.Variable(0, trainable=False)
        init_learning_rate = self.opts.learning_rate
        decay_every = self.opts.decay_every 
        decay_ratio = self.opts.decay_ratio
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                decay_every, decay_ratio, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)   
        # optimize only generative params o/w will also move the 
        # discriminator's params, which actually works, but 
        # is just incorrect      
        grads_params = opt.compute_gradients(loss, self.g_params) 
        max_norm = self.opts.max_norm
        clipped_grads_params = [(tf.clip_by_norm(g, max_norm), p) for g, p in grads_params]
        self._train_gen = opt.apply_gradients(clipped_grads_params, global_step=global_step)

    def dis_optimize(self, loss):
        global_step = tf.Variable(0, trainable=False)
        init_learning_rate = self.opts.learning_rate
        decay_every = self.opts.decay_every 
        decay_ratio = self.opts.decay_ratio
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                decay_every, decay_ratio, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)
        grads_params = opt.compute_gradients(loss, self.d_params)
        max_norm = self.opts.max_norm
        clipped_grads_params = [(tf.clip_by_norm(g, max_norm), p) 
                                        for g, p in grads_params]

        self.clipped_grads_params = clipped_grads_params

        self._train_dis = opt.apply_gradients(clipped_grads_params) 

    def optimize_baseline(self, loss):
        global_step = tf.Variable(0, trainable=False)
        init_learning_rate = self.opts.learning_rate
        decay_every = self.opts.decay_every 
        decay_ratio = self.opts.decay_ratio
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                decay_every, decay_ratio, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)
        grads_params = opt.compute_gradients(loss, self.b_params)
        self._train_baseline = opt.apply_gradients(grads_params) 

        opt = tf.train.AdamOptimizer(self.opts.learning_rate)    
        self._train_baseline = opt.minimize(loss)   

    def get_z(self):
        # pass in z noise only once at the beginning
        z_lim = self.opts.z_lim
        z = np.random.uniform(-z_lim, z_lim, size=(self.opts.batch_size, self.opts.z_dim))
        return z

    def train_generator(self):
        losses = []
        batch_probs = []
        baseline_losses = []
        num_batches = self.opts.num_samples / self.opts.batch_size

        for gepoch in range(self.opts.epoch_multiple_gen):
            for bidx in range(num_batches):
                
                # build the dict to feed inputs to graph
                feed = {}
                z = self.get_z()
                feed[self.z_placeholder] = z
                feed[self.dropout_placeholder] = self.opts.dropout
                feed[self.temperature_placeholder] = self.opts.temperature

                # perform the actual training step if training
                output_values = [self._train_gen, self.gen_train_loss_out, 
                                self.generated, self.timestep_probs, self._train_baseline, 
                                self.baseline_loss]
                _, loss_out, generated, probs, _, b_loss = self.sess.run(output_values, 
                    feed_dict=feed)
                # output_values = [self._train_gen, self.gen_train_loss_out, 
                #                 self.generated, self.timestep_probs]
                # _, loss_out, generated, probs, = self.sess.run(output_values, 
                #     feed_dict=feed)
                # b_loss = 0

                if any(np.isnan(probs.flatten())):
                    print generated
                    print z

                # add generated samples to dataset for training discriminator
                self.dataset.add_generated_samples(generated)

                losses.append(loss_out)
                batch_probs.append(probs)
                baseline_losses.append(b_loss)

        return losses, batch_probs, baseline_losses

    def train_discriminator(self):
        losses = []

        for depoch in range(self.opts.epoch_multiple_dis):
            for inputs, targets, in self.dataset.next_batch():

                # build the dict to feed inputs to graph
                feed = {}
                feed[self.labels_placeholder] = inputs
                feed[self.targets_placeholder] = targets
                feed[self.dropout_placeholder] = self.opts.dropout

                # perform the actual training step if training
                output_values = [self._train_dis, self.dis_train_loss_out, self.ce]
                _, loss_out, ce = self.sess.run(output_values, feed_dict=feed)

                if np.isnan(loss_out):
                    print inputs
                    print targets
                    for t in self.dataset.data['generated_samples']:
                        print t
                    print ce
                    raw_input()

                losses.append(loss_out)

        self.dataset.reset_generated_samples()
        self.epoch += 1
        return losses
        
    def run_epoch(self):
        # for now, track losses manually
        gen_losses = []
        dis_losses = []
        baseline_losses = []

        gen_losses_out, batch_probs, baseline_loss = self.train_generator()
        dis_losses_out = self.train_discriminator()

        gen_losses.append(gen_losses_out)
        dis_losses.append(dis_losses_out)
        baseline_losses.append(baseline_loss)

        mean_gen_loss = np.mean(gen_losses_out)
        mean_dis_loss = np.mean(dis_losses_out)
        perplexity = learning_utils.calculate_perplexity(np.array(batch_probs)[:, 1:, :])

        print('train epoch: {}\tgen loss: {}\tdis loss: {}\tperplexity: {}'.format(
            self.epoch, mean_gen_loss, mean_dis_loss, perplexity))

        self.train_gen_losses.append(mean_gen_loss)
        self.train_dis_losses.append(mean_dis_loss)
        self.baseline_losses.append(np.mean(baseline_losses))
        self.perplexities.append(perplexity)

    def build_xent_graph(self):
        self.build_xent_placeholders()
        self.xent_scores = self.build_xent_model(self.xent_inputs_placeholder)
        self.xent_loss = self.build_xent_loss(self.xent_scores)
        self._train_xent = self.build_xent_train_op(self.xent_loss)
        tf.initialize_all_variables().run()

    def build_xent_placeholders(self):
        self.xent_inputs_placeholder = tf.placeholder(tf.int32,
                                  shape=(None, self.opts.sequence_length),
                                  name="xent_input_placeholder")
        self.xent_targets_placeholder = tf.placeholder(tf.int32, 
                                  shape=(None, self.opts.sequence_length),
                                  name="xent_targets_placeholder")

    def build_xent_model(self, inputs):
        batch_size = self.opts.batch_size
        embed_dim = self.opts.embed_dim
        vocab_dim = self.dataset.vocab_dim
        z_dim = self.opts.z_dim
        num_hidden = self.opts.num_hidden
        sequence_length = self.opts.sequence_length

        with tf.variable_scope("grnn", reuse=True) as scope:
            L = tf.get_variable('L', (vocab_dim, embed_dim))
            lstm = rnn.rnn_cell.BasicLSTMCell(num_hidden)
            gWz = tf.get_variable('gWz', [z_dim, lstm.state_size])
            gbz = tf.get_variable('gbz', [lstm.state_size])
            gWo = tf.get_variable("gWo", (num_hidden, vocab_dim))
            gbo = tf.get_variable("gbo", (vocab_dim, ), initializer=tf.zeros)

        # compute initial hidden state using sampled z value
        cell_state = tf.matmul(self.z_placeholder, gWz) + gbz

        rnn_scores = []
        for idx in range(sequence_length):

            # retrieve embedding
            word_labels = inputs[:, idx]
            next_input = tf.nn.embedding_lookup(L, word_labels)
                
            with tf.variable_scope("grnn", reuse=True) as scope:
                # compute the next hidden state and cell state
                hidden_state, cell_state = lstm(next_input, cell_state)

            # project hidden state to size of vocabulary
            # no nonlinearity here because this will be feed into a softmax
            # as the scores for the words in the vocabulary
            scores = tf.matmul(hidden_state, gWo) + gbo

            # accumulate the scores
            rnn_scores.append(tf.expand_dims(scores, 1))

        # return the scores
        rnn_scores = tf.concat(values=rnn_scores, concat_dim=1)

        return rnn_scores

    def build_xent_loss(self, scores):
        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        vocab_dim = self.dataset.vocab_dim
        scores = tf.reshape(scores, (batch_size * sequence_length, vocab_dim))

        logits = [scores]
        targets = [tf.reshape(self.xent_targets_placeholder, [-1])]
        weights = [tf.ones((batch_size * sequence_length,))]
        loss = sequence_loss(logits, targets, weights)
        return loss

    def build_xent_train_op(self, loss):
        opt = tf.train.AdamOptimizer(self.opts.pretrain_learning_rate)
        train_op = opt.minimize(loss)
        return train_op

    def run_pretrain_epoch(self):
        
        losses = []
        for inputs, targets in self.dataset.next_supervised_batch():

            feed = {}
            z = self.get_z()
            feed[self.z_placeholder] = z
            feed[self.xent_inputs_placeholder] = inputs
            feed[self.xent_targets_placeholder] = targets

            output_values = [self._train_xent, self.xent_loss]
            _, xent_loss = self.sess.run(output_values, feed_dict=feed)
            losses.append(xent_loss)

        avg_xent_loss = np.mean(losses)
        print("cross entropy loss: {}".format(avg_xent_loss))
        self.pretrain_losses.append(avg_xent_loss)

    def plot_results(self):
        plt.plot(np.array(self.train_gen_losses), c='blue', linestyle='solid',
            label='training gen loss')
        plt.plot(np.array(self.train_dis_losses), c='red', linestyle='solid', 
            label='training dis loss')
        plt.legend()
        plt.title('Generator and Discriminator Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('../media/loss.png')
        plt.close()

        plt.plot(self.perplexities, c='green', label='perplexity')
        plt.legend()
        plt.savefig('../media/perplexity.png')
        plt.close()

        plt.plot(self.baseline_losses, c='teal', label='baseline loss')
        plt.legend()
        plt.savefig('../media/baseline_loss.png')
        plt.close()

        plt.plot(self.pretrain_losses, c='green', label='xent loss')
        plt.legend()
        plt.savefig('../media/xent_loss.png')
        plt.close()

        np.savez('../media/training_info.npz', gen_loss=self.train_gen_losses, 
                dis_loss=self.train_dis_losses, perplexity=self.perplexities,
                baseline_loss=self.baseline_losses, xent_loss=self.pretrain_losses)

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
        probs = []
        for batch in space:
            feed = {self.z_placeholder: batch, 
                    self.dropout_placeholder: 1,
                    self.temperature_placeholder: self.opts.sampling_temperature}
            output_values = [self.generated, self.timestep_probs]
            generated_samples, timestep_probs = self.sess.run(output_values, feed_dict=feed)

            self.dataset.add_generated_samples(generated_samples)

            samples += generated_samples.tolist()
            probs += timestep_probs.tolist()

        return np.array(samples), probs




