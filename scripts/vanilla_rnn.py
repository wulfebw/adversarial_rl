"""
vanilla rnn: given input of shape (batch_size, sequence_length, input_shape), 
    predict an output of shape (batch_size, output_shape), where input and output
    shapes can be arrays.
"""

import lasagne
import numpy as np
import sys
import theano
import theano.tensor as T

class RecurrentNetwork(object):

    def __init__(self, batch_size, sequence_length, input_shape, output_shape,
            num_hidden, learning_rate, update_rule, network_type, rng=None):
        
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.update_rule = update_rule
        self.network_type = network_type
        self.rng = rng if rng else np.random.RandomState()
        lasagne.random.set_rng(self.rng)
        self.initialize_network()
        self.update_counter = 0

    def train(self, inputs, targets):

        self.update_counter += 1

        self.inputs_shared.set_value(inputs)
        self.targets_shared.set_value(targets)

        loss = self._train()
        return loss

    def predict(self, inputs):
        self.inputs_shared.set_value(inputs)
        predictions = self._predict()
        return predictions

    def initialize_network(self):
        # unpack so easier to use
        batch_size, sequence_length = self.batch_size, self.sequence_length
        input_shape, output_shape = self.input_shape, self.output_shape 
        
        # get the function that build the network
        build_network = self.get_build_network()
        
        # build the network
        self.l_out = build_network(batch_size, sequence_length, input_shape, output_shape)

        # initialize theano symbolic variables used for compiling functions
        # sym_input shape (batch_size, sequence_length, input_shape)
        sym_inputs = T.tensor3('sym_inputs')
        # sym_target shape (batch_size, output_shape)
        sym_targets = T.matrix('sym_targets')
        
        # initialize numeric variables 
        self.input_batch_shape = (batch_size,) + (sequence_length,) + (input_shape,)
        self.output_batch_shape = (batch_size,) + (output_shape,)
        self.inputs_shared = theano.shared(np.zeros(self.input_batch_shape, dtype=theano.config.floatX))
        self.targets_shared = theano.shared(np.zeros(self.output_batch_shape, dtype=theano.config.floatX))

        # formulate the symbolic loss 
        # predicted values shape is (batch_size, output_shape)
        predictions = lasagne.layers.get_output(self.l_out, sym_inputs)

        # mean squared error cost for now
        cost = T.sum(0.5 * (predictions - sym_targets) ** 2)

        # formulate the symbolic updates 
        params = lasagne.layers.helper.get_all_params(self.l_out)
        updates = self.initialize_updates(self.update_rule, cost, params, self.learning_rate)

        # compile theano function for training 
        givens = {
            sym_inputs: self.inputs_shared,
            sym_targets: self.targets_shared
        }

        self._train = theano.function([], [cost], updates=updates, givens=givens)
        self._predict = theano.function([], [predictions], givens={sym_inputs: self.inputs_shared})

    def get_build_network(self):
        if self.network_type == 'single_layer_lstm':
            return self.build_single_layer_lstm_network
        raise ValueError("Unrecognized network_type: {}".format(self.network_type))

    def initialize_updates(self, update_rule, loss, params, learning_rate):
        if update_rule == 'adam':
            return lasagne.updates.adam(loss, params, learning_rate)
        raise ValueError("Unrecognized update: {}".format(update_rule))

    def build_single_layer_lstm_network(self, batch_size, sequence_length, input_shape, output_shape):

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
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out
