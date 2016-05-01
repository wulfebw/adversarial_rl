"""
Given an input sequence, learns to predict an output sequence.
In this case, just an autoencoder.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

NUM_SAMPLES = 10
BATCH_SIZE = 4
SEQUENCE_LENGTH = 4
INPUT_DIM = 4 # size of vocabulary
NUM_HIDDEN = 50

NUM_EPOCHS = 100

def get_random_autoencoding_data():
    X_train = np.random.randn(NUM_SAMPLES, SEQUENCE_LENGTH, INPUT_DIM)
    X_train[X_train > 0] = 1
    X_train[X_train <= 0] = -1

    # use zeros appended to sequence to indicate start decoding
    y_train = np.hstack((np.zeros((X_train.shape[0], 1, X_train.shape[2])), X_train))

    X_val = np.random.randn(NUM_SAMPLES, SEQUENCE_LENGTH, INPUT_DIM)
    X_val[X_val > 0] = 1
    X_val[X_val <= 0] = -1

    # use zeros appended to sequence to indicate start decoding
    y_val = np.hstack((np.zeros((X_val.shape[0], 1, X_val.shape[2])), X_val))

    return X_train.astype('float32'), y_train.astype('float32'), \
            X_val.astype('float32'), y_val.astype('float32')

def train_rnn_seq_to_seq():
    # get the data
    X_train, y_train, X_val, y_val = get_random_autoencoding_data()
    encoder_inputs = []
    input_feed = {}
    decoder_inputs = []

    # define the forward propagation through the network
    for idx in range(SEQUENCE_LENGTH):
        
        encoder_inputs.append(tf.placeholder(tf.float32, 
                shape=(BATCH_SIZE, INPUT_DIM), 
                name= 'encoder_inputs_{}'.format(idx)))
        input_feed[encoder_inputs[idx]] = X_train[:, idx, :]

        decoder_inputs.append(tf.placeholder(tf.float32, 
                shape=(BATCH_SIZE, INPUT_DIM), 
                name= 'decoder_inputs_{}'.format(idx)))
        input_feed[decoder_inputs[idx]] = y_train[:, idx, :]


    decoder_inputs.append(tf.placeholder(tf.float32, 
                shape=(BATCH_SIZE, INPUT_DIM), 
                name= 'decoder_inputs_{}'.format(SEQUENCE_LENGTH)))
    input_feed[decoder_inputs[SEQUENCE_LENGTH]] = y_train[:, idx, :]

    cell = rnn.rnn_cell.BasicLSTMCell(NUM_HIDDEN)
    _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=tf.float32)

    sm_w = tf.Variable(tf.random_uniform([NUM_HIDDEN, INPUT_DIM], -1, 1), name="sm_w")
    sm_b = tf.Variable(tf.zeros([INPUT_DIM]), name="sm_b")

    def loop_function(prev, i):
        return tf.matmul(prev, sm_w) + sm_b

    outputs, states = seq2seq.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loop_function) 

    predictions = []
    for idx in range(SEQUENCE_LENGTH):
        pred = tf.matmul(outputs[idx], sm_w) + sm_b
        predictions.append(pred)

    # define the loss and gradients 
    losses = []
    for idx in range(SEQUENCE_LENGTH):
        #diff = y_train[:, idx, :] - predictions[idx]
        diff = decoder_inputs[idx + 1] - predictions[idx]
        loss = tf.reduce_mean(tf.square(diff))
        losses.append(loss)

    # get cumulative loss
    loss = tf.add_n(losses)

    # define optimizer
    opt = tf.train.GradientDescentOptimizer(0.1)

    # define update
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    updates = opt.apply_gradients(zip(gradients, params))
      
    # run
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        losses = []
        for eidx in range(NUM_EPOCHS):

            batch_iterations = X_train.shape[0] / BATCH_SIZE
            assert batch_iterations == int(batch_iterations), "data must be divisible by batch size"
            batch_iterations = int(batch_iterations)

            epoch_losses = []
            for bidx in range(batch_iterations):

                start_idx = bidx * BATCH_SIZE
                end_idx = (bidx + 1) * BATCH_SIZE

                for idx in range(SEQUENCE_LENGTH):
                    input_feed[encoder_inputs[idx]] = X_train[start_idx:end_idx, idx, :]
                    input_feed[decoder_inputs[idx]] = y_train[start_idx:end_idx, idx, :]
                input_feed[decoder_inputs[SEQUENCE_LENGTH]] = y_train[start_idx:end_idx, SEQUENCE_LENGTH, :]

                _ = sess.run([updates], feed_dict=input_feed)
                loss_out = sess.run(loss, feed_dict=input_feed)
                epoch_losses.append(loss_out)

            print('epoch {} average loss: {}'.format(eidx, np.mean(epoch_losses)))
            losses.append(epoch_losses)

        plt.plot(np.array(losses).flatten())
        plt.show()

        # for _ in range(100):
        #     sess.run(updates, feed_dict=input_feed)
        # print sess.run(loss, feed_dict=input_feed)

        # # print sess.run(predictions[0], feed_dict=input_feed)
        # # print sess.run(losses[0], feed_dict=input_feed)
        # # print sess.run(loss, feed_dict=input_feed)
        # # print sess.run(opt, feed_dict=input_feed)

        # # for _ in range(100):
        # #     ups = sess.run(gradients, feed_dict=input_feed)



        # # print sess.run(loss, feed_dict=input_feed)
        # # print sess.run(params, feed_dict=input_feed)
        # # print sess.run(gradients, feed_dict=input_feed)
        # # print sess.run(params)
        # # print sess.run(updates, feed_dict=input_feed)
        # # print sess.run(params)
        # # updates, loss, pred = sess.run(output_feed, input_feed)
        # # print updates
        # # print loss
        # # print pred
        

if __name__ == '__main__':
    train_rnn_seq_to_seq()

