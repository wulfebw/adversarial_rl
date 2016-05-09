"""
Adversarial networks implementation
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, seq2seq

class AdversarialNets(object):

    def __init__(self, options, session, dataset):
        self._opts = options
        self._sess = session
        self._dataset = dataset
        self._build_placeholders()
        self._build_gen_graph()
        self._build_dis_graph()
        self.epoch = 0
        self.updates = 0
        self.train_gen_losses = []
        self.train_dis_losses = []
        self.val_gen_losses = []
        self.val_dis_losses = []

    def _build_placeholders(self):
        # unpack values for easier reference
        batch_size = self._opts.batch_size
        input_dim = self._opts.fake_input_dim
        z_dim = self._opts.z_dim
        num_hidden = self._opts.num_hidden

        # placeholder for z, randomly sampled data
        self.z = tf.placeholder(tf.float32, shape=(batch_size, z_dim), name='z')
        # placeholder for inputs
        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, input_dim), name='inputs')
        # placeholder for target values
        self.targets = tf.placeholder(tf.float32, shape=(self._opts.batch_size, 1), name='targets')

    def _build_gen_graph(self):
        # forward pass through generator
        # returns a (batch_size, input_dim) output
        self.generated = self.generate()

        # get the predictions from the discriminator
        # returns a (batch_size, 1) output
        self.gen_predictions = self.discriminate(self.generated)

        # formulate the loss
        self.gen_train_loss_out = self.gen_train_loss(self.gen_predictions)

        # create the gen train op
        self.gen_optimize(self.gen_train_loss_out)

        # also get the validation loss
        #self.gen_val_loss_out = self.gen_val_loss()

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def _build_dis_graph(self):
        # forward pass through generator to get predictions
        # returns a (batch_size, 1) output
        self.predictions = self.discriminate(self.inputs)

        # get the loss value
        self.dis_train_loss_out = self.dis_train_loss(self.predictions)

        # create the dis train op
        self.dis_optimize(self.dis_train_loss_out)

        # also get the validation loss
        #self.dis_val_loss_out = self.dis_val_loss()

        # initialize all variable and prep to save model
        tf.initialize_all_variables().run()

    def generate(self):
        # unpack values for easier reference
        batch_size = self._opts.batch_size
        input_dim = self._opts.fake_input_dim
        z_dim = self._opts.z_dim
        num_hidden = self._opts.num_hidden

        # network
        # input-to-hidden
        gW1 = tf.get_variable('gW1', [z_dim, num_hidden])
        gb1 = tf.get_variable('gb1', [num_hidden])
        h = tf.nn.tanh(tf.matmul(self.z, gW1) + gb1)

        # hidden-to-output
        gW2 = tf.get_variable('gW2', [num_hidden, input_dim])
        gb2 = tf.get_variable('gb2', [input_dim])
        self.generated = tf.nn.tanh(tf.matmul(h, gW2) + gb2)

        return self.generated

    def discriminate(self, inputs, trainable=True):
        # unpack values for easier reference
        batch_size = self._opts.batch_size
        input_dim = self._opts.fake_input_dim
        num_hidden = self._opts.num_hidden

        # input-to-hidden
        dW1 = tf.get_variable('dW1', [input_dim, num_hidden])
        db1 = tf.get_variable('db1', [num_hidden])
        h = tf.nn.tanh(tf.matmul(inputs, dW1) + db1)

        # hidden-to-output
        dW2 = tf.get_variable('dW2', [num_hidden, 1])
        db2 = tf.get_variable('db2', [1])
        predictions = tf.nn.tanh(tf.matmul(h, dW2) + db2)

        return predictions

    def gen_train_loss(self, predictions):
        return tf.nn.sigmoid_cross_entropy_with_logits(predictions, tf.ones_like(predictions))

    def dis_train_loss(self, predictions):
        return tf.nn.sigmoid_cross_entropy_with_logits(predictions, self.targets)

    def gen_val_loss(self, predictions):
        pass

    def dis_val_loss(self, predictions):
        pass

    def gen_optimize(self, loss):
        # define optimizer
        opt = tf.train.AdamOptimizer(self._opts.learning_rate)

        # add reg
        tf.get_variable_scope().reuse_variables()
        gW1 = tf.get_variable('gW1')
        gW2 = tf.get_variable('gW2')
        reg_loss = self._opts.reg_scale * (tf.nn.l2_loss(gW1) + tf.nn.l2_loss(gW2))

        self._train_gen = opt.minimize(reg_loss + loss)

    def dis_optimize(self, loss):
        # define optimizer
        opt = tf.train.AdamOptimizer(self._opts.learning_rate)

        # add reg
        tf.get_variable_scope().reuse_variables()
        dW1 = tf.get_variable('dW1')
        dW2 = tf.get_variable('dW2')
        reg_loss = self._opts.reg_scale * (tf.nn.l2_loss(dW1) + tf.nn.l2_loss(dW2))

        self._train_dis = opt.minimize(reg_loss + loss)

    def get_z(self):
        z = np.random.uniform(-1, 1, size=(self._opts.batch_size, self._opts.z_dim))
        return z

    def train_generator(self, validation=False):
        losses = []
        num_batches = self._opts.fake_num_samples / self._opts.batch_size
        for bidx in range(num_batches):
            
            # build the dict to feed inputs to graph
            feed = {}
            z = self.get_z()
            feed[self.z] = z

            if validation:
                # in validation just get the predictions
                loss_out = self._sess.run([self.gen_val_loss_out], feed_dict=feed)
            else: 
                # perform the actual training step if training
                output_values = [self._train_gen, self.gen_train_loss_out, self.generated]
                _, loss_out, generated = self._sess.run(output_values, feed_dict=feed)

            self._dataset.add_generated_samples(generated)
            losses.append(loss_out)

        return losses

    def train_discriminator(self, validation=False):
        losses = []

        for inputs, targets, in self._dataset.next_batch():

            # build the dict to feed inputs to graph
            feed = {}
            feed[self.inputs] = inputs
            feed[self.targets] = targets

            if validation:
                # in validation just get the predictions
                loss_out = self._sess.run([self.dis_val_loss_out], feed_dict=feed)
            else: 
                # perform the actual training step if training
                output_values = [self._train_dis, self.dis_train_loss_out]
                _, loss_out = self._sess.run(output_values, feed_dict=feed)
            
            losses.append(loss_out)

        self.epoch += 1
        return losses
        
    def run_epoch(self, validation=False):
        # for now, track losses manually
        gen_losses = []
        dis_losses = []

        gen_losses_out = self.train_generator(validation)
        gen_losses.append(gen_losses_out)
        dis_losses_out = self.train_discriminator(validation)
        dis_losses.append(dis_losses_out)

        mean_gen_loss = np.mean(gen_losses_out)
        mean_dis_loss = np.mean(dis_losses_out)

        if validation:
            print('val epoch: {}\tgen loss: {}\tdis loss: {}'.format(self.epoch, 
                    mean_gen_loss, mean_dis_loss))
            self.val_gen_losses.append(mean_gen_loss)
            self.val_dis_losses.append(mean_dis_loss)
        else:
            print('train epoch: {}\tgen loss: {}\tdis loss: {}'.format(self.epoch, 
                    mean_gen_loss, mean_dis_loss))
            self.train_gen_losses.append(mean_gen_loss)
            self.train_dis_losses.append(mean_dis_loss)

    def plot_results(self):
        plt.plot(np.array(self.train_gen_losses), c='blue', linestyle='solid',
            label='training gen loss')
        plt.plot(np.array(self.train_dis_losses), c='red', linestyle='solid', 
            label='training dis loss')
        plt.plot(np.array(self.val_gen_losses), c='blue', linestyle='dashdot',
            label='validation gen loss')
        plt.plot(np.array(self.val_dis_losses), c='red', linestyle='dashdot', 
            label='validation dis loss')
        plt.legend()
        plt.show()

    def sample_space(self):
        """
        Generates samples linearly from z space.
        """
        batch_size = self._opts.batch_size
        steps = 5 * batch_size
        space = np.linspace(-1, 1, steps).reshape(-1, batch_size, 1)
        samples = []
        for batch in space:
            feed = {self.z: batch}
            generated_samples = self._sess.run([self.generated], feed_dict=feed)
            samples += generated_samples[0].tolist()

        return np.array(samples)
            




