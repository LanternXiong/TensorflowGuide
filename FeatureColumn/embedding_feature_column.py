import tensorflow as tf
from tensorflow.python import feature_column
from tensorflow.python import pywrap_tensorflow


def run_embedding_feature_column():
    example = {'example': [['A'], ['B'], ['C'], ['D'], ['E'], ['F'], ['G'], ['H'], ['I'], ['J']]}
    print(example['example'])
    example_column = feature_column.categorical_column_with_hash_bucket('example', hash_bucket_size=15)
    example_column_embedding = feature_column.embedding_column(example_column, dimension=4)
    example_transformed_tensor = feature_column.input_layer(example, [example_column_embedding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('Result:')
        print(session.run([example_transformed_tensor]))


if __name__ == '__main__':
    run_embedding_feature_column()