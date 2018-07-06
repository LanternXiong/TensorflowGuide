import tensorflow as tf
import metadata


def parse_csv(raw_csv_line, is_serving=False):
    columns = tf.decode_csv(raw_csv_line, record_defaults=metadata.HEADER_DEFAULTS)
    features = dict(zip(metadata.HEADER, columns))
    labels = features.pop(metadata.LABEL_NAME)
    if is_serving:
        return features
    return features, labels


def input_fn(file_name_list, mode=tf.estimator.ModeKeys.TRAIN, num_epochs=5, batch_size=10):
    def _input_fn():
        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
        buffer_size = 2 * batch_size + 1
        dataset = tf.data.TextLineDataset(file_name_list)
        dataset = dataset.map(parse_csv)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()
    return _input_fn


def serving_input_fn():
    csv_line = tf.placeholder(shape=[None], dtype=tf.string)

    features = parse_csv(csv_line, is_serving=True)

    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors={'csv_line': csv_line})

