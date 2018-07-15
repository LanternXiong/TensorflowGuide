import tensorflow as tf
from tensorflow.python import feature_column


def run_bucketized_feature_column():
    price = {'price': [[1., 1., 1., 1., 1., 1.], [2., 2., 2., 2., 2., 2.], [3., 3., 3., 3., 3., 3.], [4., 4., 4., 4., 4., 4.]]}
    print(price['price'])
    price_column = feature_column.numeric_column('price', shape=[6])
    bucket_price = feature_column.bucketized_column(price_column, [0, 2, 3.5, 5])
    price_transformed_tensor = feature_column.input_layer(price, [bucket_price])

    with tf.Session() as session:
        print('Result:')
        print(session.run([price_transformed_tensor]))


if __name__ == '__main__':
    run_bucketized_feature_column()
