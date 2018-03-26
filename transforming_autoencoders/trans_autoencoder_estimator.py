import tensorflow as tf

def create_transforming_autoencoder(model_dir,
                                    num_capsules,
                                    num_rec,
                                    num_gen,
                                    trans_size):
    feature_columns = [ tf.feature_column.numeric_column('image'),
                        tf.feature_column.numeric_column('transformation')]

    return tf.estimator.Estimator(model_dir=model_dir,
                model_fn=model_fn,
                params={
                    'feature_columns': feature_columns,
                    'num_capsules': num_capsules,
                    'num_rec': num_rec,
                    'num_gen': num_gen,
                    'trans_size': trans_size
                })

def model_fn(features, labels, mode, params):
    in_image = features['image']
    in_trans = features['transformation']

    outputs = []
    for i in range(params['num_capsules']):
        outputs.append(capsule(in_image, in_trans,
                               n_rec=params['num_rec'],
                               n_gen=params['num_gen'],
                               trans_size=params['trans_size'],
                               name='capsule-%d'%i))

    out_image = tf.add_n(outputs, name='out_image')

    # Compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'out_image': out_image
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss
    loss = tf.losses.mean_squared_error(labels=labels, predictions=out_image)

    # Compute evaluation metrics
    mean_squared_error = tf.metrics.mean_squared_error(labels=labels,
                                                predictions=out_image,
                                                name='mean_squared_error_op')
    metrics = {'mean_squared_error': mean_squared_error}
    tf.summary.scalar('mean_squared_error', mean_squared_error[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    # Add optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



def capsule(input_img, input_trans, n_rec, n_gen, trans_size, name='capsule'):
    with tf.variable_scope(name) as scope:
        flat = tf.layers.flatten(input_img,
                                 name='flatten')

        # Recognation units
        r = tf.layers.dense(flat,
                            units=n_rec,
                            activation=tf.nn.sigmoid,
                            name='recognation-units')

        # Probability that the entity is present
        p = tf.layers.dense(r,
                            units=1,
                            activation=tf.nn.sigmoid,
                            name='probability')

        # Transformation
        t = tf.layers.dense(r,
                            units=trans_size,
                            activation=None,
                            name='transformation-prediction')

        # Output transformation
        t_out = tf.add(t, input_trans,
                       name='transformation-output')


        # Generation units
        g = tf.layers.dense(t_out,
                            units=n_gen,
                            activation=tf.nn.sigmoid,
                            name='generation-units')

        # Transformed output
        out_flat = tf.layers.dense(g,
                                   units=flat.shape[1],
                                   activation=None,
                                   name='transformed-output')

        out_flat = tf.Print(out_flat, [tf.shape(out_flat)])

        out = tf.reshape(out_flat, tf.shape(input_img),
                         name='reshape')

        # Scaled output TODO make rank dependent on rank of image
        return tf.multiply(out, tf.reshape(p, (-1,1,1,1)),
                           name='scaled-output')
