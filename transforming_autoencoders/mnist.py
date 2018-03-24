import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from transforming_autoencoders import data

class MNIST:
    """A class for loading the MNIST dataset and providing input functions on it."""


    def __init__(self, directory):
        self.mnist = input_data.read_data_sets(directory)

    def train_input_fn(self):
        return lambda x, min_trans, max_trans: data.train_input_fn(self.mnist.train.images, x, 28, 28, min_trans, max_trans)

    def eval_input_fn(self):
        return lambda x, min_trans, max_trans: data.eval_input_fn(self.mnist.test.images, x, 28, 28, min_trans, max_trans)

    def train_size(self):
        return len(self.mnist.train.images)

    def eval_size(self):
        return len(self.mnist.test.images)
