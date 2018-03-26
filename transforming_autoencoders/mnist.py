import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from transforming_autoencoders import data

class MNIST:
    """A class for loading the MNIST dataset and providing input functions on
    it.
    """


    def __init__(self, directory):
        self.mnist = input_data.read_data_sets(directory)

    def train_input_fn(self):
        def fn(batch_size, transformer_fn, parallel_calls):
            return data.train_input_fn(self.mnist.train.images,
                                batch_size,
                                transformer_fn,
                                lambda i: data.reshape_with_channel(i, 28, 28),
                                parallel_calls)
        return fn

    def eval_input_fn(self):
        def fn(batch_size, transformer_fn, parallel_calls):
            return data.eval_input_fn(self.mnist.train.images,
                                batch_size,
                                transformer_fn,
                                lambda i: data.reshape_with_channel(i, 28, 28),
                                parallel_calls)
        return fn

    def train_size(self):
        return len(self.mnist.train.images)

    def eval_size(self):
        return len(self.mnist.test.images)
