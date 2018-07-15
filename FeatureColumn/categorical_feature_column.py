import tensorflow as tf
from tensorflow.python import feature_column


def run_categorical_hash_bucket_feature_column():
    example = {'example': [['A'], ['B'], ['C'], ['D']]}
    print(example['example'])
    example_column = feature_column.categorical_column_with_hash_bucket('example', hash_bucket_size=7)
    example_column_identy = feature_column.indicator_column(example_column)
    example_transformed_tensor = feature_column.input_layer(example, [example_column_identy])

    with tf.Session() as session:
        print('Result:')
        print(session.run([example_transformed_tensor]))


def run_categorical_vocabulary_list_feature_column():
    example = {'example': [['A'], ['B'], ['C'], ['D']]}
    print(example['example'])
    example_column = feature_column.categorical_column_with_vocabulary_list('example', vocabulary_list=['A', 'D'])
    example_column_identy = feature_column.indicator_column(example_column)
    example_transformed_tensor = feature_column.input_layer(example, [example_column_identy])

    with tf.Session() as session:
        # session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('Result:')
        print(session.run([example_transformed_tensor]))


def run_categorical_identity_feature_column():
    example = {'example': [[1], [2], [8], [4]]}
    print(example['example'])
    example_column = feature_column.categorical_column_with_identity('example', num_buckets=10)
    example_column_identity = feature_column.indicator_column(example_column)
    example_transformed_tensor = feature_column.input_layer(example, [example_column_identity])

    with tf.Session() as session:
        # session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('Result:')
        print(session.run([example_transformed_tensor]))


if __name__ == '__main__':

    run_categorical_identity_feature_column()
