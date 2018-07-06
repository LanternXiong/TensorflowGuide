import tensorflow as tf
import metadata


def create_feature_columns():
    feature_columns = {feature_name: tf.feature_column.numeric_column(feature_name, normalizer_fn=None)
                       for feature_name in metadata.INPUT_NUMERIC_FEATURE_NAMES}
    return feature_columns
