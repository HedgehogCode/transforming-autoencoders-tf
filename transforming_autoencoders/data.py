import tensorflow as tf

def transformer_map(img, transformer_fn):
    trans_in, trans_out, trans_delta = transformer_fn()
    return ({'image': tf.contrib.image.transform(img, trans_in),
             'transformation': trans_delta },
            tf.contrib.image.transform(img, trans_out))

# TODO remove min_trans and use -max_trans
def translation_fn(batch_size, min_trans=-2, max_trans=2):
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
    return trans_in, trans_out, trans_delta

def affine_fn(batch_size, stddev, max_trans):
    zeros = tf.zeros((batch_size,))

    def random_affine_matrix():
        # Generate random affine transformations
        a0 = tf.random_normal((batch_size,), mean=1.0, stddev=stddev)
        a1 = tf.random_normal((batch_size,), mean=0.0, stddev=stddev)
        a2 = tf.random_uniform((batch_size,),
                minval=-max_trans, maxval=max_trans)
        b0 = tf.random_normal((batch_size,), mean=0.0, stddev=stddev)
        b1 = tf.random_normal((batch_size,), mean=1.0, stddev=stddev)
        b2 = tf.random_uniform((batch_size,),
                minval=-max_trans, maxval=max_trans)
        return tf.stack([a0,a1,a2,b0,b1,b2,zeros,zeros],axis=1)

    trans_in = random_affine_matrix()
    trans_out = random_affine_matrix()

    ones = tf.ones((batch_size,1))
    a = tf.reshape(tf.concat([trans_in, ones], axis=1), (batch_size,3,3))
    b = tf.reshape(tf.concat([trans_out, ones], axis=1), (batch_size,3,3))
    ainv = tf.matrix_inverse(a)
    d = tf.matmul(b,ainv)
    d_flat,norm = tf.split(tf.reshape(d, (batch_size,9)), [8,1], 1)
    trans_delta = tf.divide(d_flat, norm)

    return trans_in, trans_out, trans_delta

def reshape_with_channel(inp, width, height):
    return tf.reshape(inp, (-1, width, height, 1))

def train_input_fn(images,
                   batch_size,
                   transformer_fn,
                   reshape_fn,
                   parallel_calls=8):
    """An input function for training an autoencoder.

    Args:
        images (numpy array): Array of images.
        batch_size (int): The size of one batch for training.
        transformer_fn (fun): Function which creates ops which create random
            transformations.
        reshape_fn (fun): Function which takes an input and returns an op which
            reshapes the input. If the images array contains images with the
            correct shape this function can also be the identity.
        parallel_calls (int): Number of parallel_calls for the map function of
            the Dataset. The number of available CPU cores should be a good
            value.
    """
    d = tf.data.Dataset.from_tensor_slices((images,))
    d = d.cache()
    d = d.repeat()
    d = d.shuffle(60000)
    d = d.batch(batch_size)
    d = d.map(reshape_fn, num_parallel_calls=parallel_calls)
    d = d.map(lambda x: transformer_map(x, transformer_fn),
              num_parallel_calls=parallel_calls)
    d = d.prefetch(buffer_size=batch_size)
    return d.make_one_shot_iterator().get_next()

def eval_input_fn(images,
                  batch_size,
                  transformer_fn,
                  reshape_fn,
                  parallel_calls=8):
    """An input function for evaluating an autoencoder.

    Args:
        images (numpy array): Array of images.
        batch_size (int): The size of one batch for training.
        transformer_fn (fun): Function which creates ops which create random
            transformations.
        reshape_fn (fun): Function which takes an input and returns an op which
            reshapes the input. If the images array contains images with the
            correct shape this function can also be the identity.
        parallel_calls (int): Number of parallel_calls for the map function of
            the Dataset. The number of available CPU cores should be a good
            value.
    """
    d = tf.data.Dataset.from_tensor_slices((images,))
    d = d.batch(batch_size)
    d = d.map(reshape_fn, num_parallel_calls=parallel_calls)
    d = d.map(lambda x: transformer_map(x, transformer_fn),
              num_parallel_calls=parallel_calls)
    return d.make_one_shot_iterator().get_next()
