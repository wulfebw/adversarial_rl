"""
Train adversarial networks. Note that training these things is way
more nuanced than is training a normal network. 

Interestingly, if you don't use logs in the loss functions, still works?

"""

import matplotlib.pyplot as plt
import numpy as np

import adversarial_rnn
import training_data

BATCH_SIZE = 2

def run(network, data, num_epochs):
    X_train, X_val = data['X_train'], data['X_val']

    dis_losses = []
    gen_losses = []
    for eidx in range(num_epochs):

        batch_iterations = X_train.shape[0] / BATCH_SIZE
        assert batch_iterations == int(batch_iterations), "data must be divisible by batch size"
        batch_iterations = int(batch_iterations)

        epoch_fake_losses = []
        for bidx in range(batch_iterations):

            start_idx = bidx * BATCH_SIZE
            end_idx = (bidx + 1) * BATCH_SIZE
            inputs = X_train[start_idx:end_idx]

            dis_loss, gen_loss = network.train(inputs)
            dis_losses.append(dis_loss)
            gen_losses.append(gen_loss)

    print(network.generate(X_val))
    plt.plot(np.array(dis_losses), c='red')
    plt.plot(np.array(gen_losses), c='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss during training')
    plt.show()

if __name__ == '__main__':
    data = training_data.get_xor_data_unsupervised()
    X_train = data['X_train']
    batch_size = BATCH_SIZE
    sequence_length = X_train.shape[1]
    input_shape = X_train.shape[2]

    network = adversarial_rnn.AdversarialNetwork(BATCH_SIZE, sequence_length, 
                input_shape, num_hidden=50, learning_rate=0.001)
    run(network, data, num_epochs=10000)

