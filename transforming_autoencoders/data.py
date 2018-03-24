import tensorflow as tf

# TODO make more general for other transformations
def transformer_fn(inp, w, h, min_trans=-2, max_trans=2):
    # Reshape the image
    img = tf.reshape(inp, (-1, w, h, 1))
    batch_size = tf.shape(img)[0]

    # Generatre random numbers for the transformation
    x1 = tf.random_uniform((batch_size,), minval=min_trans, maxval=max_trans)
    y1 = tf.random_uniform((batch_size,), minval=min_trans, maxval=max_trans)
    x2 = tf.random_uniform((batch_size,), minval=min_trans, maxval=max_trans)
    y2 = tf.random_uniform((batch_size,), minval=min_trans, maxval=max_trans)


    # Format the input and output transformation
    ones = tf.ones((batch_size,))
    zeros = tf.zeros((batch_size,))
    trans_in =  tf.stack([ones , zeros, x1,
                          zeros, ones , y1,
                          zeros, zeros], axis=1)
    trans_out = tf.stack([ones , zeros, x2,
                          zeros, ones , y2,
                          zeros, zeros], axis=1)
    trans_delta = tf.subtract(
                            tf.stack([x1, y1], axis=1),
                            tf.stack([x2, y2], axis=1))

    return ({'image': tf.contrib.image.transform(img, trans_in),
             'transformation': trans_delta },
            tf.contrib.image.transform(img, trans_out))


def train_input_fn(images, batch_size, width, height, min_trans=-2, max_trans=2, parallel_calls=8):
    """An input function for training an autoencoder"""
    d = tf.data.Dataset.from_tensor_slices((images,))
    d = d.cache()
    d = d.repeat()
    d = d.shuffle(60000)
    d = d.batch(batch_size)
    d = d.map(lambda x: transformer_fn(x, width, height, min_trans, max_trans),
              num_parallel_calls=parallel_calls)
    d = d.prefetch(buffer_size=batch_size)
    return d.make_one_shot_iterator().get_next()

def eval_input_fn(images, batch_size, width, height, min_trans=-2, max_trans=2, parallel_calls=8):
    """An input function for evaluating an autoencoder"""
    d = tf.data.Dataset.from_tensor_slices((images,))
    d = d.batch(batch_size)
    d = d.map(lambda x: transformer_fn(x, width, height, min_trans, max_trans),
              num_parallel_calls=parallel_calls)
    return d.make_one_shot_iterator().get_next()
