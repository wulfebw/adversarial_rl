
import numpy as np
import tensorflow as tf


def batch_sample_with_temperature_old(arr, temperature=1.0):
    """
    Samples from something resembeling a multinomial distribution.
    Works by multiplying the probabilities of each value by a 
    random uniform number and then selecting the max.

    Where arr is of shape (batch_size, vocab_size)
    Returns the index of the item that was sampled in each row.

    source: https://github.com/tensorflow/tensorflow/issues/456
    """
    batch_size, vocab_size = arr.get_shape()

    with tf.op_scope([arr, temperature], "batch_sample_with_temperature"):


        # subtract by the largest value in each batch to improve stability
        c = tf.reduce_max(arr, reduction_indices=1, keep_dims=True)
        softmax = tf.nn.softmax(arr - c) + 1e-6
        x = tf.log(softmax) # / temperature

        # softmax again
        x = tf.nn.softmax(x) / temperature

        # perform the sampling
        u = tf.random_uniform(tf.shape(arr), minval=1e-6, maxval=1)
        sampled_idx = tf.argmax(tf.sub(x, -tf.log(-tf.log(u))), dimension=1) 
        
    return sampled_idx, x

def batch_sample_with_temperature(a, temperature=1.0):
    new_scores = a / temperature 
    matrix_X = tf.nn.softmax(new_scores)
    matrix_U = tf.random_uniform(tf.shape(a), minval = 0, maxval = 1)
    final_number = tf.argmax(tf.log(matrix_X)-tf.log(-tf.log(matrix_U)),dimension=1)
    return final_number, matrix_X

def reduce_std(arr, mean):
    """
    Seems like tf doesn't have a reduce_std?
    so implement that here, reducing over all 
    but the last dimension
    """
    indices = range(len(arr.get_shape()) - 1)
    std = tf.sqrt(tf.reduce_mean((arr - mean) ** 2, reduction_indices=indices))
    return std

def get_std_mean(batch):
    """
    Find the std dev and mean across the batch for each input 
    value (i.e., for each element of an input)

    So if the shape of this input is (10, 5, 2), this
    returns a tuple, the first element being the std dev
    of shape (2) and the second being the mean also shape (2).

    Reduction performed over all dimensions but the last.
    """
    indices = range(len(batch.get_shape()) - 1)
    batch_mean = tf.reduce_mean(batch, reduction_indices=indices)
    batch_std = reduce_std(batch, batch_mean)
    return batch_std, batch_mean

def calculate_perplexity(probs):

    shape = np.shape(probs)
    total_size = 1
    for size in shape:
        total_size *= size

    perp = -np.sum(np.log(probs)) / total_size
    return np.exp(perp)
    
