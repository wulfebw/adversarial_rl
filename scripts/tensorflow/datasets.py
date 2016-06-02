
import collections
import csv
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATASET = "title"
TITLE_FILEPATH = '/Users/wulfebw/Dropbox/School/Stanford/spring_2016/cs224d/project/data/game_summarization/{}_words.csv'.format(DATASET)
TWITCH_FILEPATH = '../data/chat.in'

def mlb_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    yield (x, y)

def get_mlb_dataset(limit=-1):
    input_filepath = TITLE_FILEPATH
    count = 0
    word_count = 0
    with open(input_filepath, 'rb') as infile:
        csvreader = csv.reader(infile, delimiter=' ')
        for row in csvreader:
            count += 1
            for word in row:
                yield word
            yield '<eos>'

            if limit > 0 and count > limit:
                raise StopIteration

def load_twitch_dataset(limit):
    input_filepath = TWITCH_FILEPATH
    count = 0
    sentence_count = 0
    with open(input_filepath, 'rb') as infile:
        for row in infile:
            words = row.strip().split(' ')

            if sentence_count > limit:
                raise StopIteration
            sentence_count += 1

            for word in words:
                yield word
            yield '<eos>'

class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_counts = collections.defaultdict(int)
    self.word_freq = collections.defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.add_word(self.unknown, count=0)

  def add_word(self, word, threshold=0, count=1):
    if self.word_counts[word] < threshold:
      return

    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
  
  def limited_construct(self, words):
    count_words, words = itertools.tee(words)
    for word in count_words:
      self.word_counts[word] += 1
    for word in words:
      self.add_word(word, threshold=2)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

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
        self.epoch = 0

    def next_batch(self, validation=False):
        """
        Returns a batch of data consisting of both "real" data from the 
        original dataset created during initialization as well as data
        from the generative model.
        """

        # get the data
        X = self.data["X_val"] if validation else self.data["X_train"] 
        generated_samples = self.data["generated_samples"]

        # decide how many batches there will be
        # because each batch is half real and half fake
        # the number of batches will be 2 times the number
        # of batches in the smaller collection
        num_samples = min(len(X), len(generated_samples))
        num_batches = (2 * num_samples) / self._opts.batch_size 

        # go through all the batches collecting real and 
        # fake data and yielding it
        for bidx in range(num_batches):
            # effective batch size is real batch_size / 2
            effective_batch_size = self._opts.batch_size / 2
            # start and end indices
            start = bidx * effective_batch_size
            end = (bidx + 1) * effective_batch_size

            # real data
            real_x_batch = X[start:end]
            real_y = np.ones((effective_batch_size, 1))

            # fake data
            fake_x_batch = generated_samples[start:end]
            fake_y = np.zeros((effective_batch_size, 1))

            # combine the two
            inputs = np.vstack((real_x_batch, fake_x_batch))
            targets = np.vstack((real_y, fake_y))

            yield inputs, targets

        # if reset:
        #     self.reset_generated_samples()

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

        Parameters:
        - samples is of shape (batch_size, input_dim)
        """
        if 'generated_samples' not in self.data:
            self.data['generated_samples'] = []
        assert samples.shape == (self._opts.batch_size, self._opts.fake_input_dim)

        if len(self.data['generated_samples']) == 0:
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
        np.random.seed(1)
        data = {}
        num_samples = self._opts.fake_num_samples

        # X_train = self._make_normal_dataset()
        # X_train = self._make_one_zero_dataset()
        X_train = self._make_square_dataset()
        data['X_train'] = X_train
        
        # X_val = np.tile([1,0], (num_samples / self._opts.fake_val_ratio, 1))
        # data['X_val'] = X_val
        self.data = data

    def _make_one_zero_dataset(self):
        num_samples = self._opts.fake_num_samples
        return np.tile([1,0], (num_samples, 1))

    def _make_normal_dataset(self):
        """
        best hyperparams:
        opts.learning_rate = .001
        opts.train_ratio = 2
        opts.fake_num_samples = 128
        opts.epochs_to_train = 200
        opts.num_hidden = 100
        opts.z_dim = 2
        opts.reg_scale = 0
        opts.dropout = 1
        """
        num_samples = self._opts.fake_num_samples
        mean = 0
        std = 1

        def get_normal(x):
            return 1 / math.sqrt(math.pi * 2) * math.exp(-(x - mean) ** 2 / (2 * std ** 2))

        xs = np.linspace(-4, 4, num_samples)
        inputs = np.array([[x, get_normal(x)] for x in xs])
        # permuting is really important
        permuted_inputs = np.random.permutation(inputs)
        return permuted_inputs

    def _make_square_dataset(self):
        """
        best hyperparams:
        opts.learning_rate = .00008 
        opts.train_ratio = 5 
        opts.fake_num_samples = 32 
        opts.epochs_to_train = 300 
        opts.num_hidden = 200 
        opts.z_dim = 2
        opts.reg_scale = 0 
        opts.dropout = 1
        """
        num_samples = self._opts.fake_num_samples
        # do it by edges of square:
        side_length = 5
        edge = np.linspace(0, side_length, num_samples/4)
        bot = np.array([np.array([e, 0]) for e in edge])
        top = np.array([np.array([e, side_length]) for e in edge])
        left = np.array([np.array([0, e]) for e in edge])
        right = np.array([np.array([side_length, e]) for e in edge])
        inputs = np.vstack((bot, top, left, right))
        permuted_inputs = np.random.permutation(inputs)
        return permuted_inputs

class FakeRecurrentAdversarialDataset(object):

    def __init__(self, opts):
        self._opts = opts
        self._make_fake_dataset()
        self.epoch = 0

    def next_batch(self):
        """
        Returns a batch of data consisting of both "real" data from the 
        original dataset created during initialization as well as data
        from the generative model.
        """

        # get the data
        X = self.data["X_train"] 
        generated_samples = self.data["generated_samples"]

        # decide how many batches there will be
        # because each batch is half real and half fake
        # the number of batches will be 2 times the number
        # of batches in the smaller collection
        num_samples = min(len(X), len(generated_samples))
        num_batches = (2 * num_samples) / self._opts.batch_size 

        # go through all the batches collecting real and 
        # fake data and yielding it
        for bidx in range(num_batches):

            # effective batch size is real batch_size / 2
            effective_batch_size = self._opts.batch_size / 2
            # start and end indices
            start = bidx * effective_batch_size
            end = (bidx + 1) * effective_batch_size

            # real data
            real_x_batch = X[start:end]
            real_y = np.ones((effective_batch_size, 1))

            # fake data
            fake_x_batch = generated_samples[start:end]
            fake_y = np.zeros((effective_batch_size, 1))

            # combine the two
            inputs = np.vstack((real_x_batch, fake_x_batch))
            targets = np.vstack((real_y, fake_y))

            yield inputs, targets

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

        Parameters:
        - samples is of shape (batch_size, input_dim)
        """

        if 'generated_samples' not in self.data:
            self.data['generated_samples'] = []
        assert samples.shape == (self._opts.batch_size, self._opts.sequence_length, self._opts.input_dim) or samples.shape == (self._opts.batch_size, self._opts.sequence_length)

        if len(self.data['generated_samples']) > self._opts.num_samples / 2:
            return

        if len(self.data['generated_samples']) == 0:
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
        np.random.seed(1)
        data = {}
        num_samples = self._opts.num_samples

        if self._opts.dataset_name == 'sine':
            X_train = self._make_sine_dataset()
        elif self._opts.dataset_name == 'circle':
            X_train = self._make_circle_dataset()
        elif self._opts.dataset_name == 'alphabet':
            X_train = self._make_alphabet_dataset()
        elif self._opts.dataset_name == 'mlb':
            X_train = self._make_mlb_dataset()
        elif self._opts.dataset_name == 'twitch':
            X_train = self._make_twitch_dataset()
        else:
            raise ValueError("invalid dataset name: {}".format(self._opts.dataset_name))
        data['X_train'] = X_train
        data['generated_samples'] = []
        self.data = data

    def _make_sine_dataset(self):
        """
        make a sin dataset with num_samples, each of sequence_length, where
        each timestep of the seqeunce contains two elements:
        x: just the x position
        y: the sine of x

        so z is of shape (num_samples, sequence_length, 2), and one example
        would be [[0, 0], [pi/2, 1], [3/4pi, -1], [2pi, 0]]
        """
        num_samples = self._opts.num_samples
        sequence_length = self._opts.sequence_length
        stretch = 1
        x = np.linspace(-2 * np.pi * stretch, 2 * np.pi * stretch, num_samples * sequence_length)
        y = np.sin(x / stretch)
        z = np.vstack((x, y)).T.reshape(num_samples, sequence_length, 2)
        z = np.random.permutation(z)
        return z

    def _make_circle_dataset(self):
        """
        This is a dataset consisting of subsequences of a circles in counterclockwise order.
        """
        radius = 1
        origin = (0, 0)
        dataset = np.zeros((self._opts.num_samples * self._opts.sequence_length, 2))
        angles = np.linspace(0, 2 * np.pi, self._opts.num_samples * self._opts.sequence_length)
        for idx, angle in enumerate(angles):
            x = np.cos(angle) * radius + origin[0]
            y = np.sin(angle) * radius + origin[1]
            dataset[idx, :] = [x, y]
        dataset = dataset.reshape(self._opts.num_samples, self._opts.sequence_length, 2)
        dataset = np.random.permutation(dataset)
        return dataset

    def _make_alphabet_dataset(self):
        """
        This is a discrete dataset consisting of the in-order
        alphabet in length two sequences.

        The following settings get 99% in sample for the full alphabet:
        opts = TestOptions()    
        opts.learning_rate = .01
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 5
        opts.batch_size = 52
        opts.num_samples = 130
        opts.epochs_to_train = 1000
        opts.num_hidden = 128
        opts.embed_dim = 32
        opts.z_dim = 2
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .1
        opts.full_sequence_optimization = True
        opts.save_every = 200
        opts.plot_every = 50
        opts.reduce_temperature_every = 20
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 25
        opts.decay_ratio = .96
        opts.max_norm = 5.0
        """
        assert self._opts.sequence_length == 2 
        self.vocab_dim = 26
        assert self._opts.num_samples / self.vocab_dim > 0
        self.label_to_word_dict = {}

        alphabet = []
        for ch in range(self.vocab_dim):
            self.label_to_word_dict[ch] = chr(ch + ord('a'))
            alphabet.append(ch)

        X_train = []
        for ch1, ch2 in zip(alphabet, alphabet[1:] + [alphabet[0]]):
            X_train.append([ch1, ch2])
        X_train = np.tile(X_train, (self._opts.num_samples / self.vocab_dim, 1))  
        X_train = np.random.permutation(X_train)
        return X_train

    def decode(self, label):
        """
        Given the label of a word, return that word.
        """
        if label not in self.label_to_word_dict:
            raise(ValueError("invalid label: {}".format(label)))
        return self.label_to_word_dict[label]

    def decode_dataset(self, dataset, real=False):
        """
        Decode the entire fake dataset generated by the model  
        """
        decoder = self.decode
        if real:
            decoder = self.vocab.decode

        return [[decoder(label) for label in seq] for seq in dataset]

    def fake_mlb_generator(self):
        words = """craig kimbrel downs d-backs , sets braves saves mark 
            russell martin punctuates 8-run sixth as pirates stomp brewers 
            roark fans 11 , holds padres to 3 hits in 6-0 win 
            gordon 's 2 triples lead dodgers past rockies 7-2"""
        words = words.strip().split()
        for w in words:
            yield w

    def _make_mlb_dataset(self):
        """
        seems to work:
        opts = TestOptions()    
        opts.learning_rate = .01
        opts.epoch_multiple_gen = 1
        opts.epoch_multiple_dis = 10
        opts.batch_size = 32
        opts.sequence_length = 2
        opts.num_samples = 512
        opts.epochs_to_train = 500
        opts.num_hidden = 128
        opts.embed_dim = 40
        opts.z_dim = 10
        opts.dropout = 1.
        opts.temperature = 1.
        opts.sampling_temperature = .1
        opts.full_sequence_optimization = True
        opts.save_every = 50
        opts.plot_every = 50
        opts.reduce_temperature_every = 10
        opts.temperature_reduction_amount = .01
        opts.min_temperature = .1
        opts.decay_every = 20
        opts.decay_ratio = .96
        opts.max_norm = 2.0
        opts.sentence_limit = 1
        opts.dataset_name = 'mlb'
        """
        sequence_length = self._opts.sequence_length
        num_samples = self._opts.num_samples
        limit = self._opts.sentence_limit

        self.vocab = Vocab()
        # self.vocab.construct(get_mlb_dataset(limit))
        self.vocab.construct(self.fake_mlb_generator())
        self.vocab_dim = len(self.vocab.word_to_index)
        # encoded_train = np.array([self.vocab.encode(word) 
        #                     for word in get_mlb_dataset(limit)], dtype=np.int32)
        encoded_train = np.array([self.vocab.encode(word) 
                            for word in self.fake_mlb_generator()], dtype=np.int32)


        remove = len(encoded_train) % sequence_length
        if remove > 0:
            encoded_train = encoded_train[:-remove]
        #print encoded_train
        X_train = encoded_train.reshape(-1, sequence_length)
        #print self.decode_dataset(X_train, real=True)
        #print self.vocab.word_to_index
        return X_train

    def _make_twitch_dataset(self):
        sequence_length = self._opts.sequence_length
        num_samples = self._opts.num_samples
        limit = self._opts.sentence_limit

        self.vocab = Vocab()
        self.vocab.limited_construct(load_twitch_dataset(limit))
        self.vocab_dim = len(self.vocab.word_to_index)
        encoded_train = np.array([self.vocab.encode(word) 
                            for word in load_twitch_dataset(limit)], dtype=np.int32)

        remove = len(encoded_train) % sequence_length
        if remove > 0:
            encoded_train = encoded_train[:-remove]
        X_train = encoded_train.reshape(-1, sequence_length)
        self._opts.num_samples = len(X_train)
        return X_train

    def next_supervised_batch(self):

        # get the data
        X = self.data["X_train"] 

        # decide number of batches
        num_samples = len(X)
        batch_size = self._opts.batch_size
        num_batches = (num_samples - 1) / batch_size

        # go through all the batches collecting real and 
        # fake data and yielding it
        for bidx in range(num_batches):

            # start and end indices
            start = bidx * batch_size
            end = (bidx + 1) * batch_size

            # real data
            inputs = X[start:end]
            targets = X[start + 1:end + 1]

            yield inputs, targets



