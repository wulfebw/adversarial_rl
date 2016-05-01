
import numpy as np

class FakeSeqToSeqDataSet(object):

    def __init__(self, opts):
        self._opts = opts
        self._make_fake_dataset()
        self._index_in_epoch = 0
        self.epoch = 0

    def next_batch(self, validation=False):
        """
        Returns the next batch of data, a (inputs, targets) tuple
        When all the data has been iterated through, 
        this increments its epoch count, which is used 
        by the caller to know when the epoch has finished.
        """
        end_epoch_index = self._opts.fake_num_samples
        if validation:
            end_epoch_index /= self._opts.fake_val_ratio

        # set normal start value
        start = self._index_in_epoch

        # increment the index to use as an end value
        self._index_in_epoch += self._opts.batch_size

        if self._index_in_epoch > end_epoch_index:
            # finished epoch
            self.epoch += 1
            start = 0
            self._index_in_epoch = self._opts.batch_size
            assert self._opts.batch_size < end_epoch_index

        end = self._index_in_epoch

        # return different sets depending on train / validation
        if validation:
            X = self.data['X_val'][start:end]
            y = self.data['y_val'][start:end]
        else:
            X = self.data['X_train'][start:end]
            y = self.data['y_train'][start:end]

        return X, y

    def _make_fake_dataset(self):
        """
        Makes a fake dataset consisting of random ones and negative ones.
        Each training example has input shape (sequence length, input dimension),
        and output shape (sequence length + 1, input dimension).

        As such, this fake data is autoencoding data. The target values have been
        shifted forward in time 1 step so that the initial value feed into the 
        decoding rnn of the model will start with a "<START>" value.

        In the actual model, the offset data (y_train and y_val) will be feed into
        the computational graph as encoder_inputs, but will also be used in 
        formulating the loss. To do this, we make targets[t] equal to 
        encoder_inputs[t + 1].
        """

        data = {}
        num_samples = self._opts.fake_num_samples
        seq_length = self._opts.fake_sequence_length
        input_dim = self._opts.fake_input_dim

        X_train = np.random.randn(num_samples, seq_length, input_dim)
        X_train[X_train > 0] = 1
        X_train[X_train <= 0] = -1
        data['X_train'] = X_train

        # use zeros appended to sequence to indicate start decoding
        y_train = np.hstack((np.zeros((X_train.shape[0], 1, X_train.shape[2])), X_train))
        data['y_train'] = y_train

        X_val = np.random.randn(num_samples / 10, seq_length, input_dim)
        X_val[X_val > 0] = 1
        X_val[X_val <= 0] = -1
        data['X_val'] = X_val

        # use zeros appended to sequence to indicate start decoding
        y_val = np.hstack((np.zeros((X_val.shape[0], 1, X_val.shape[2])), X_val))
        data['y_val'] = y_val

        self.data = data
