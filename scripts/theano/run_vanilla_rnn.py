"""
Train a vanilla rnn to predict a value from a sequence where the predicted value 
can only be of sequence length one, but it can be an array (e.g., the next state).
"""

import matplotlib.pyplot as plt
import numpy as np

import vanilla_rnn
import training_data

BATCH_SIZE = 2

def run(network, data, num_epochs):
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    losses = []
    for eidx in range(num_epochs):

        batch_iterations = X_train.shape[0] / BATCH_SIZE
        assert batch_iterations == int(batch_iterations), "data must be divisible by batch size"
        batch_iterations = int(batch_iterations)

        epoch_losses = []
        for bidx in range(batch_iterations):

            start_idx = bidx * BATCH_SIZE
            end_idx = (bidx + 1) * BATCH_SIZE
            inputs = X_train[start_idx:end_idx]
            targets = y_train[start_idx:end_idx]

            loss = network.train(inputs, targets)
            epoch_losses.append(loss)

        print('epoch {} average loss: {}'.format(eidx, np.mean(epoch_losses)))
        losses.append(epoch_losses)
        
    preds = network.predict(X_val)[0]
    print('validation results:')
    for pred, target in zip(np.array(preds), y_val):
        print('target: {}\tprediction: {}'.format(target, pred))

    plt.plot(np.array(losses).flatten())
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss during training')
    plt.show()

if __name__ == '__main__':
    data = training_data.get_xor_data_supervised()
    X_train, y_train = data['X_train'], data['y_train']
    batch_size = BATCH_SIZE
    sequence_length = X_train.shape[1]
    input_shape = X_train.shape[2]
    output_shape = y_train.shape[1]

    network = vanilla_rnn.RecurrentNetwork(batch_size, sequence_length, input_shape, output_shape,
                num_hidden=5, learning_rate=0.1, update_rule='adam', network_type='single_layer_lstm')
    run(network, data, num_epochs=100)








