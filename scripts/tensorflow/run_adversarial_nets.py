
import numpy as np
import tensorflow as tf

import adversarial_nets
import datasets

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to save model")
flags.DEFINE_string("train_data", None, "Training text file.")
flags.DEFINE_string("val_data", None, "Validation text file.")
flags.DEFINE_integer("epochs_to_train", 45, "Number of epochs to train.")
flags.DEFINE_float("learning_rate", 0.05, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 2, "Number of training examples per batch.")
flags.DEFINE_integer("statistics_interval", 5, "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5, "Save training summary every n seconds.")
flags.DEFINE_integer("snapshot_interval", 600, "Snapshot the model every n seconds.")
flags.DEFINE_integer("num_hidden", 25, "Number of hidden units in each layer.")
flags.DEFINE_float("grad_clip", 1., "Value to clip gradients to.")
flags.DEFINE_float("reg_scale", 0., "Regularization constant.")
flags.DEFINE_boolean("run_val", False, "Whether to run validation.")
flags.DEFINE_integer("fake_num_samples", 10, "Number of fake samples to make.")
flags.DEFINE_integer("fake_sequence_length", 10, "Sequence length of fake examples.")
flags.DEFINE_integer("fake_input_dim", 2, "Input dimension of fake examples.")
flags.DEFINE_integer("fake_val_ratio", 2, "How many times fewer samples in fake val.")


FLAGS = flags.FLAGS

class Options(object):
  """Options used by the adversrial nnet model."""

  def __init__(self):

    # Where to write out summaries and snapshots.
    self.save_path = FLAGS.save_path

    # The training text file.
    self.train_data = FLAGS.train_data

    # The text file for validation
    self.val_data = FLAGS.val_data

    # Number of epochs to train. 
    self.epochs_to_train = FLAGS.epochs_to_train

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints
    self.snapshot_interval = FLAGS.snapshot_interval

    # value to clip gradients to
    self.grad_clip = FLAGS.grad_clip

    # scaling constant for l2 penalty
    self.reg_scale = FLAGS.reg_scale

    # whether or not to run valdiation epochs
    self.run_validation = FLAGS.run_val

    # number of hidden units in the rnn layers
    self.num_hidden = FLAGS.num_hidden

    # number of fake examples to make
    self.fake_num_samples = FLAGS.fake_num_samples

    # sequence length of fake examples
    self.fake_sequence_length = FLAGS.fake_sequence_length

    # input dim of fake examples
    self.fake_input_dim = FLAGS.fake_input_dim

    # how many times fewer samples in the fake validation set than train set
    self.fake_val_ratio = FLAGS.fake_val_ratio

def main(_):

    # must have train and validation data and must have snapshot save dir
    if not FLAGS.train_data or not FLAGS.val_data or not FLAGS.save_path:
        print("If --train_data --val_data or --save_path are not specified "
            "this script will use fake data.")

    # create the options and the filename to save the model as 
    opts = Options()

    # set up session
    with tf.Graph().as_default(), tf.Session() as session:

        # load data into a DataSet object
        dataset = datasets.FakeAdversarialDataset(opts)

        # create the model
        model = adversarial_nets.AdversarialNets(opts, session, dataset)

        # run it! wooo
        for _ in xrange(opts.epochs_to_train):
            model.run_epoch()  
            if opts.run_validation:
                model.run_epoch(validation=True) 

        model.plot_results()

if __name__ == "__main__":
    # this just tells the flags above to parse the command line args
    # and then calls main(_) with the original argv values
    # which we don't care about in this case, hence the underscore
    tf.app.run()
