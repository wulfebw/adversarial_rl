
from collections import defaultdict
from copy import deepcopy
import csv
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import time

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

def get_mlb_dataset():
    input_filepath = '/Users/wulfebw/Dropbox/School/Stanford/spring_2016/cs224d/project/data/game_summarization/title_words.csv'
    with open(input_filepath, 'rb') as infile:
        csvreader = csv.reader(infile, delimiter=' ')
        for row in csvreader:
            for word in row:
                yield word
            yield '<eos>'

def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def calculate_perplexity(log_probs):
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))

class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.add_word(self.unknown, count=0)

  def add_word(self, word, count=1):
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

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

class Config(object):
  batch_size =16
  embed_size = 35
  hidden_size = 100
  num_steps = 15
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM(object):

  def __init__(self, config):
    self.config = config
    self.load_data(debug=True)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)

  def load_data(self, debug=False):
    self.vocab = Vocab()
    self.vocab.construct(get_mlb_dataset())
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_mlb_dataset()], dtype=np.int32)
    if debug:
      num_debug = 2048
      self.encoded_train = self.encoded_train[:num_debug]

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32,
                                  shape=(None, self.config.num_steps),
                                  name="input_placeholder")
    self.labels_placeholder = tf.placeholder(tf.int32, 
                                  shape=(None, self.config.num_steps),
                                  name="labels_placeholder")
    self.dropout_placeholder = tf.placeholder(tf.float32,
                                  shape=(),
                                  name="dropout_placeholder")
  
  def add_embedding(self):
    with tf.device('/cpu:0'):
      L = tf.get_variable("L", (len(self.vocab), self.config.embed_size))
      embeddings = tf.nn.embedding_lookup(L, self.input_placeholder)
      inputs = [tf.squeeze(embeds, squeeze_dims=[1]) for embeds in 
                          tf.split(1, self.config.num_steps, embeddings)]
      return inputs

  def add_projection(self, rnn_outputs):
    U = tf.get_variable("U", (self.config.hidden_size, len(self.vocab)))
    b_2 = tf.get_variable("b_2", (len(self.vocab),), initializer=tf.zeros)
    outputs = []
    for scores in rnn_outputs:
      z = tf.matmul(scores, U) + b_2
      outputs.append(z)
    return outputs

  def add_loss_op(self, output):
    logits = [output]
    targets = [tf.reshape(self.labels_placeholder, [-1])]
    weights = [tf.ones((self.config.batch_size * self.config.num_steps,))]
    loss = sequence_loss(logits, targets, weights)
    return loss

  def add_training_op(self, loss):
    opt = tf.train.AdamOptimizer(self.config.lr)
    train_op = opt.minimize(loss)
    return train_op

  def add_model(self, inputs):
    self.initial_state = tf.zeros((self.config.batch_size, self.config.hidden_size))
    prev_h = self.initial_state
    rnn_outputs = []
    for idx, embeddings in enumerate(inputs):
      with tf.variable_scope("rnn") as scope:
        if idx != 0:
          scope.reuse_variables()

        I = tf.get_variable("I", (self.config.embed_size, self.config.hidden_size))
        H = tf.get_variable("H", (self.config.hidden_size, self.config.hidden_size))
        b_1 = tf.get_variable("b_1", (self.config.hidden_size, ), initializer=tf.zeros)

      drop_embeddings = tf.nn.dropout(embeddings, self.dropout_placeholder)
      z = tf.matmul(drop_embeddings, I) + tf.matmul(prev_h, H) + b_1
      prev_h = tf.sigmoid(z)
      drop_prev_h = tf.nn.dropout(prev_h, self.dropout_placeholder)
      rnn_outputs.append(drop_prev_h)

    self.final_state = rnn_outputs[-1]
    return rnn_outputs

  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in mlb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      mlb_iterator(data, config.batch_size, config.num_steps)):
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=15, stop_tokens=None, temp=1.0):
  state = model.initial_state.eval()
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(stop_length):
    x = np.reshape(tokens[-1], (-1, 1))
    feed_dict = {model.input_placeholder: x,
                  model.initial_state: state,
                  model.dropout_placeholder: 1}
    y_pred, state = session.run([model.predictions[-1], model.final_state], feed_dict=feed_dict)
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_rnnlm():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM(config)
    scope.reuse_variables()
    gen_model = RNNLM(gen_config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:  
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)

      print 'Training perplexity: {}'.format(train_pp)

    saver.save(session, '../../snapshots/mlb_rnnlm.weights')
    saver.restore(session, '../../snapshots/mlb_rnnlm.weights')
    starting_text = 'astros'
    while starting_text:
      print ' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_rnnlm()
