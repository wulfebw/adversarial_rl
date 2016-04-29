"""
adversarial rnn: orchestrates generative and discriminative rnns.

generative rnn: given input of shape (batch_size, sequence_length, input_shape), 
    generate an output of shape (batch_size, input_shape), where input  
    shape can be array and the output is a prediction of the next value.

discriminative rnn: given input of shape (batch_size, sequence_length + 1, input_shape), 
    predict an output of shape (batch_size, 1), where each element of the output is a 
    probability value that the input to the network was real.
"""

import collections
import lasagne
import numpy as np
import sys
import theano
import theano.tensor as T

class AdversarialNetwork(object):

    def __init__(self, batch_size, sequence_length, input_shape, num_hidden, learning_rate):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.initialize_networks()
        self.update_counter = 0

    def train(self, inputs):
        self.update_counter += 1

        # real train
        self.inputs_shared.set_value(inputs)
        real_targets = np.ones((inputs.shape[0], 1))
        self.targets_shared.set_value(real_targets)
        real_loss = self._real_train()

        # fake train
        truncated_inputs = inputs[:, :-1, :]
        self.inputs_shared.set_value(truncated_inputs)
        fake_targets = np.zeros((inputs.shape[0], 1))
        self.targets_shared.set_value(fake_targets)
        dis_loss, gen_loss = self._fake_train()

        return dis_loss, gen_loss

    def generate(self, inputs):
        truncated_inputs = inputs[:, :-1, :]
        self.inputs_shared.set_value(truncated_inputs)
        predictions = self._generate()
        return predictions

    def initialize_networks(self):
        # unpack so easier to use
        batch_size = self.batch_size
        sequence_length = self.sequence_length
        input_shape = self.input_shape 
        
        # build the network
        # gen_out shape is (batch_size, sequence_length + 1, input_shape)
        self.gen_out = self.build_gen_network(batch_size, sequence_length, input_shape)
        # dis_out shape is (batch_size, 1)
        self.dis_out = self.build_dis_network(batch_size, sequence_length, input_shape)

        # initialize theano symbolic variables used for compiling functions
        # sym_input shape (batch_size, sequence_length, input_shape)
        sym_inputs = T.tensor3('sym_inputs')
        # sym_target shape (batch_size)
        sym_targets = T.matrix('sym_targets')
        
        # initialize numeric variables 
        self.input_batch_shape = (batch_size,) + (sequence_length,) + (input_shape,)
        self.inputs_shared = theano.shared(np.zeros(self.input_batch_shape, dtype=theano.config.floatX))
        self.targets_shared = theano.shared(np.zeros((self.batch_size, 1), dtype=theano.config.floatX))
        # two training functions, one for fake data and one for real data

        ### fake train ###
        # formulate the symbolic loss 
        # predicted values shape is (batch_size, output_shape)
        generated_sequences = lasagne.layers.get_output(self.gen_out, sym_inputs)

        # discriminative prediction
        predictions = lasagne.layers.get_output(self.dis_out, generated_sequences)

        # cross entropy
        def m_binary_cross_entropy(preds, targets):
            return T.mean(-(targets * T.log(preds) + (1.0 - targets) * T.log(1.0 - preds)))

        dis_cost = T.mean(predictions) 
        gen_cost = T.mean(1 - predictions) 
        # dis_cost = m_binary_cross_entropy(predictions, sym_targets)
        # gen_cost = T.mean(T.log(1 - predictions))

        # formulate the symbolic updates 
        dis_params = lasagne.layers.helper.get_all_params(self.dis_out)
        dis_updates = lasagne.updates.adam(dis_cost, dis_params, self.learning_rate)

        gen_params = lasagne.layers.helper.get_all_params(self.gen_out)
        gen_updates = lasagne.updates.adam(gen_cost, gen_params, self.learning_rate)

        updates = dis_updates.copy()
        updates.update(gen_updates)

        # compile theano function for training 
        givens = {
            sym_inputs: self.inputs_shared,
            # sym_targets: self.targets_shared
        }

        self._fake_train = theano.function([], [dis_cost, gen_cost], updates=updates, givens=givens)
        self._generate = theano.function([], [generated_sequences], givens={sym_inputs: self.inputs_shared})
        
        ### real train ###
        # formulate the symbolic loss

        # discriminative prediction
        predictions = lasagne.layers.get_output(self.dis_out, sym_inputs)

        # cross entropy
        cost = m_binary_cross_entropy(predictions, sym_targets)

        # formulate the symbolic updates 
        dis_params = lasagne.layers.helper.get_all_params(self.dis_out)
        dis_updates = lasagne.updates.adam(cost, dis_params, self.learning_rate)

        # compile theano function for training 
        givens = {
            sym_inputs: self.inputs_shared,
            sym_targets: self.targets_shared
        }

        self._real_train = theano.function([], [cost], updates=dis_updates, givens=givens)

    def build_dis_network(self, batch_size, sequence_length, input_shape):

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, input_shape)
        )

        default_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.))
        forget_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(1.))
        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=1,
            only_return_final=True,
            name='l_lstm1'
        )

        l_out = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out

    def build_gen_network(self, batch_size, sequence_length, input_shape):

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, input_shape)
        )

        default_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.))
        forget_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(1.))
        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=1,
            only_return_final=True,
            name='l_lstm1'
        )

        l_output = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=input_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        l_shaped = lasagne.layers.ReshapeLayer(l_output, (batch_size, 1, input_shape))
        l_out = lasagne.layers.ConcatLayer([l_in, l_shaped], name='l_merge_out')

        return l_out