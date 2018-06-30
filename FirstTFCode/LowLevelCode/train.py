import os

import tensorflow as tf
from inference import model_inference
from conf import data_conf, model_conf, training_conf
from data_input import read_csv_input_data


def train():
    # here to get input data, `None` in `shape` means that this dimension depends on(is equal to) batch size
    x_input = tf.placeholder(dtype=tf.float32, shape=[None, model_conf['layer_conf']['input_node']], name='x-input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, model_conf['layer_conf']['output_node']], name='y-input')

    # define regularizer # optional
    regularizer = tf.contrib.layers.l2_regularizer(training_conf['regularization_rate'])

    # feed forward
    y = model_inference(x_input, regularizer)

    # use cross entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # cross_entropy loss + regularizer loss
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # use gradient descent to optimize the model
    train_op = tf.train.GradientDescentOptimizer(training_conf['learning_rate'])

    # minimize loss
    train_step = train_op.minimize(loss)

    saver = tf.train.Saver()

    batch_features, batch_labels = read_csv_input_data(data_conf['data_input_path'],
                                                       record_defaults=data_conf['record_defaults'],
                                                       batch_size=training_conf['batch_size'],
                                                       num_epochs=training_conf['num_epochs'])
    with tf.Session() as sess:
        # initialize all variables //because we use `initializer=tf.truncated_normal_initializer(stddev=0.1)` in
        # `get_weight_variable`
        tf.global_variables_initializer().run()

        # queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        local_init_op = tf.local_variables_initializer()
        for i in range(training_conf['training_steps']):
            sess.run(local_init_op)
            # here we get real value to feed dict
            batch_features_value, batch_labels_value = sess.run([batch_features, batch_labels])
            _, loss_value = sess.run([train_step, loss],
                                     feed_dict={x_input: batch_features_value, y_input: batch_labels_value})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
                # save the model
                saver.save(sess, os.path.join(model_conf['model_save_path'], 'iris_model'))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
