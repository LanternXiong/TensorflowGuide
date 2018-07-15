import tensorflow as tf
from tensorflow.python import feature_column


def my_normalizer_fn(x):
    return x * 2.0


def run_numeric_feature_column(normalizer_fn=None):
    price = {'price': [[1., 1., 1., 1., 1., 1.], [2., 2., 2., 2., 2., 2.], [3., 3., 3., 3., 3., 3.], [4., 4., 4., 4., 4., 4.]]}
    print(price['price'])
    price_column = feature_column.numeric_column('price', shape=[6], normalizer_fn=normalizer_fn)
    price_transformed_tensor = feature_column.input_layer(price, [price_column])

    with tf.Session() as session:
        print('Result:')
        print(session.run([price_transformed_tensor]))


if __name__ == '__main__':
    run_numeric_feature_column(my_normalizer_fn)
