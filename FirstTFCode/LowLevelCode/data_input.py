import tensorflow as tf


def read_csv_input_data(file_path_list, record_defaults, batch_size=60, num_epochs=None, shuffle=True):
    '''

    :param file_path_list: A 1-D string tensor with the strings to produce (python list with strings is also ok)
    :param record_defaults:A list of `Tensor` objects with specific types.
      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
    :param batch_size:The batch size pulled from the queue.
    :param num_epochs:repeat files in file_path_list num_epoch times
    :param shuffle:shuffle the files in file_path_list
    :return:
    '''
    # produce a queue to read the file in file_path list
    file_name_queue = tf.train.string_input_producer(file_path_list, num_epochs=num_epochs, shuffle=shuffle)
    # get each line in the file
    reader = tf.TextLineReader()
    _, value = reader.read(file_name_queue)
    # decode each line
    decode_result = tf.decode_csv(value, record_defaults=record_defaults)
    # get features and labels
    features = decode_result[:-1]
    label = decode_result[-1:]
    # get a batch of data //The `capacity` argument controls the how long the prefetching is allowed to
    #   grow the queues.
    batch_features, batch_labels = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity=5000, min_after_dequeue=1000)
    # batch_features, batch_labels = tf.train.batch([features, label], batch_size=batch_size)  # not shuffle
    return batch_features, batch_labels


def read_test():
    file_path_list = ['../Data/iris_training.csv']
    record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0]]
    example, label = read_csv_input_data(file_path_list, record_defaults=record_defaults)
    # Note: if `num_epochs` is not `None` in `string_input_producer`, `string_input_producer` creates local counter
    #   `epochs`. Use `local_variables_initializer()` to initialize local variables.
    local_init_op = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(local_init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(3):
            print(sess.run([example, label]))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    read_test()
