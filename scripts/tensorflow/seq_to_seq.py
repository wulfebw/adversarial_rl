"""
An encoder-decoder rnn implementation. Maps sequences of input values to sequences of 
output values. 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

class SeqToSeq(object):

    def __init__(self, options, session, dataset):
        self._opts = options
        self._sess = session
        self._dataset = dataset
        self._build_graph()
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []

    def _build_graph(self):
        # forward pass to get predictions
        predictions = self.forward()

        # get the loss value, which is its own function 
        # because it could be many things in the future
        self.train_loss_out = self.loss(predictions)

        # get the updates and store them because they 
        # allow for training
        self.updates = self.optimize(self.train_loss_out)

        # also get the validation loss
        self.val_loss_out = self.val_loss()

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def forward(self):
        # unpack values for easier reference
        seq_length = self._opts.fake_sequence_length
        batch_size = self._opts.batch_size
        input_dim = self._opts.fake_input_dim
        num_hidden = self._opts.num_hidden

        # define the placeholders / symbolic inputs to the graph
        encoder_inputs = []
        decoder_inputs = []
        for idx in range(seq_length):
        
            encoder_inputs.append(tf.placeholder(tf.float32, 
                shape=(batch_size, input_dim), 
                name= 'encoder_inputs_{}'.format(idx)))
            
            decoder_inputs.append(tf.placeholder(tf.float32, 
                shape=(batch_size, input_dim), 
                name= 'decoder_inputs_{}'.format(idx)))

        # we do this an extra time for the decoder because
        # has a <START> token appended to the front
        decoder_inputs.append(tf.placeholder(tf.float32, 
            shape=(batch_size, input_dim), 
            name= 'decoder_inputs_{}'.format(seq_length)))
        

        # create the encoder rnn
        self.cell = rnn.rnn_cell.BasicLSTMCell(num_hidden)
        _, self.enc_state = rnn.rnn(self.cell, encoder_inputs, dtype=tf.float32)

        # define a custom function to convert each decoder output 
        # at each timestep of dimension num_hidden into 
        # the same dimension as the output (which in this case
        # is the same as the input) so it can be used as a prediction
        # or as the input to the next time step of the decoding
        self.W_out = tf.Variable(tf.random_uniform([num_hidden, input_dim], -1, 1), name="sm_w")
        self.b_out = tf.Variable(tf.zeros([input_dim]), name="sm_b")
        def loop_function(prev, i):
            return tf.matmul(prev, self.W_out) + self.b_out

        # build the decoder rnn
        outputs, states = seq2seq.rnn_decoder(decoder_inputs, self.enc_state, self.cell) 

        # so the outputs are the scores
        # we could convert them to probability distributions
        # with a softmax, but for now just treat them as the 
        # direct predictions
        predictions = []
        for idx in range(seq_length):
            pred = loop_function(outputs[idx], idx)
            predictions.append(pred)


        # set the encoder_inputs and decoder_inputs to be members 
        # of the object because they are required for each train step
        # in contrast, the predictions are only used for defining
        # the graph, so we just return them once
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs

        return predictions

    def loss(self, predictions):
        # unpack values for easier reference
        seq_length = self._opts.fake_sequence_length

        # the targets are the same as the decoder_inputs
        # except shifted ahead in time 1 unit
        targets = [dec_input for dec_input in self.decoder_inputs[1:]]

        # compute the loss, which for now is squared error
        # this could also be cross entropy and, in fact, probably should be
        losses = []
        for idx in range(seq_length):
            diff = targets[idx] - predictions[idx]
            loss = tf.reduce_mean(tf.square(diff))
            losses.append(loss)

        # get and return cumulative loss
        loss = tf.add_n(losses)
        return loss

    def val_loss(self):
        # reuse vars 
        tf.get_variable_scope().reuse_variables()

        # unpack values for easier reference
        seq_length = self._opts.fake_sequence_length

        def loop_function(prev, i):
            return tf.matmul(prev, self.W_out) + self.b_out

        # build the decoder rnn
        outputs, states = seq2seq.rnn_decoder(self.decoder_inputs, 
                            self.enc_state, self.cell, loop_function) 

        # so the outputs are the scores
        # we could convert them to probability distributions
        # with a softmax, but for now just treat them as the 
        # direct predictions
        predictions = []
        for idx in range(seq_length):
            pred = loop_function(outputs[idx], idx)
            predictions.append(pred)

        # the targets are the same as the decoder_inputs
        # except shifted ahead in time 1 unit
        targets = [dec_input for dec_input in self.decoder_inputs[1:]]

        # compute the loss, which for now is squared error
        losses = []
        for idx in range(seq_length):
            diff = targets[idx] - predictions[idx]
            loss = tf.reduce_mean(tf.square(diff))
            losses.append(loss)

        # get and return cumulative loss
        loss = tf.add_n(losses)
        return loss

    def optimize(self, loss):
        # define optimizer
        opt = tf.train.AdamOptimizer(self._opts.learning_rate)

        # get params
        params = tf.trainable_variables()

        # add regularization
        reg_loss = tf.Variable(0., 'reg_loss')
        for p in params:
            reg_loss += tf.nn.l2_loss(p)
        loss += reg_loss * self._opts.reg_scale

        # define update
        gradients = tf.gradients(loss, params)
        max_gradient_norm = self._opts.grad_clip
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        updates = opt.apply_gradients(zip(clipped_gradients, params))

        # return the updates, which allow for the actual training
        return updates

    def run_epoch(self, validation=False):
        # unpack for easier use
        seq_length = self._opts.fake_sequence_length

        # for now, track losses manually
        losses = []

        # loop until we have gone through the full dataset once
        while self._dataset.epoch == self.epoch:

            # sample a batch
            inputs, targets = self._dataset.next_batch(validation)

            # build the dict to feed correct inputs to graph
            feed = {}
            for idx in range(seq_length):

                # this sets the enc / dec inputs to the appropriate
                # timestep of the inputs / targets
                feed[self.encoder_inputs[idx]] = inputs[:, idx, :]
                feed[self.decoder_inputs[idx]] = targets[:, idx, :]

            # again do an extra time for the decoder
            feed[self.decoder_inputs[seq_length]] = targets[:, seq_length, :]

            if validation:
                # in validation just get the predictions
                loss_out = self._sess.run([self.val_loss_out], feed_dict=feed)
            else: 
                # perform the actual training step if training
                _, loss_out = self._sess.run([self.updates, self.train_loss_out], feed_dict=feed)
            
            losses.append(loss_out)

        if validation:
            print('val epoch: {} average loss: {}'.format(self.epoch - 1, np.mean(losses)))
            self.epoch += 1
            self.val_losses.append(np.mean(losses))
        else:
            print('train epoch: {} average loss: {}'.format(self.epoch, np.mean(losses)))
            self.epoch += 1
            self.train_losses.append(np.mean(losses))

    def plot_results(self):
        plt.plot(np.array(self.train_losses), c='red', label='training loss')
        plt.plot(np.array(self.val_losses), c='blue', label='val loss')
        plt.legend()
        plt.show()



