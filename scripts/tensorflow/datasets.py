
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

        X_val = np.random.randn(num_samples / self._opts.fake_val_ratio, seq_length, input_dim)
        X_val[X_val > 0] = 1
        X_val[X_val <= 0] = -1
        data['X_val'] = X_val

        # use zeros appended to sequence to indicate start decoding
        y_val = np.hstack((np.zeros((X_val.shape[0], 1, X_val.shape[2])), X_val))
        data['y_val'] = y_val

        self.data = data

class FakeAdversarialDataset(object):

    def __init__(self, opts):
        self._opts = opts
        self._make_fake_dataset()
        self._index_in_epoch = 0
        self.epoch = 0

    def next_batch(self, validation=False):
        """
        Returns a batch of data consisting of both "real" data from the 
        original dataset created during initialization as well as data
        from the generative model from this epoch only.
        """
        end_epoch_index = self._opts.fake_num_samples
        if validation:
            end_epoch_index /= self._opts.fake_val_ratio

        # set normal start value
        start = self._index_in_epoch

        # increment the index to use as an end value
        self._index_in_epoch += self._opts.batch_size / 2

        reset_generated_samples = False
        if self._index_in_epoch > end_epoch_index:
            # finished epoch
            self.epoch += 1

            start = 0
            self._index_in_epoch = self._opts.batch_size / 2
            assert self._opts.batch_size < end_epoch_index
            reset_generated_samples = True

        end = self._index_in_epoch

        # return different sets depending on train / validation
        if validation:
            real_data = self.data['X_val'][start:end]
        else:
            real_data = self.data['X_train'][start:end]

        # regardless of train vs val, use generated samples
        assert len(self.data['generated_samples']) >= end
        gen_data = self.data['generated_samples'][start:end]

        # combine and order randomly to make learning smoother
        # the data here is of shape (batch_size, num_samples)
        X = np.vstack((real_data, gen_data))
        num_labels_each = self._opts.batch_size / 2
        y = np.vstack((np.tile([1,0], (num_labels_each, 1)), np.tile([0,1], (num_labels_each, 1))))

        idxs = np.random.permutation(np.arange(len(X)))
        X = X[idxs]
        y = y[idxs]

        if reset_generated_samples:
            self.reset_generated_samples()

        return X, y

    def add_generated_samples(self, samples):
        """
        Training proceeds in two steps. First, we train G by generating samples
        and having D provide a reward for them. We store these samples for
        later use in training D, which is the purpose of this function - it
        stores the generated samples in order to use them for trainin D. 

        The second step in training is to train D, which, as stated, uses these
        stored, generated values, but it also uses the fake dataset created 
        during initialization.

        The generated samples list is cleared after each epoch.
        """
        if 'generated_samples' not in self.data:
            self.data['generated_samples'] = []
        assert samples.shape == (self._opts.batch_size, self._opts.fake_input_dim)

        if self.data['generated_samples'] == []:
            self.data['generated_samples'] = samples
        else:
            self.data['generated_samples'] = np.vstack((self.data['generated_samples'], samples))

    def reset_generated_samples(self):
        """
        Resets the list of generated data. 
        """
        self.data['generated_samples'] = []

    def _make_fake_dataset(self):
        """
        Makes a fake dataset consisting of discrete valued data. The dataset
        above also does this, but it was treated like real valued data. This
        dataset is also much simpler in that it consists of samples from a 
        two word vocabulary ([1,0] and [0,1]), where all those samples are
        of the first word ([1,0]).

        The goal is see if the generator network can learn to generate this 
        first word ([1,0]) from noise data. It should obviously be able to 
        do this by just setting the bias terms and ignoring the noise. What is
        of interest is the training mechanism, discussed in the file containing
        the adversarial network implementation.

        Note that this is not a sequential dataset for now. Each example is just
        a single word.
        """
        data = {}
        num_samples = self._opts.fake_num_samples

        X_train = np.tile([1,0], (num_samples, 1))
        data['X_train'] = X_train
        X_val = np.tile([1,0], (num_samples / self._opts.fake_val_ratio, 1))
        data['X_val'] = X_val
        self.data = data

    def make_fake_generated(self):
        """
        For debugging, mock the generated dataset with a fake one.
        """
        data = {}
        num_samples = self._opts.fake_num_samples

        X_train = np.tile([.5,.5], (num_samples, 1))
        self.data['generated_samples'] = X_train





