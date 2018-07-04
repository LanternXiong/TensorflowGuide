import tensorflow as tf
from conf import model_conf

INPUT_NODE = model_conf['layer_conf']['input_node']
HIDDEN_LAYER_NODE = model_conf['layer_conf']['hidden_node']
OUTPUT_NODE = model_conf['layer_conf']['output_node']


def get_weight_variable(shape, regularizer=None):
    """
    To initialize and get weights in each layer
    :param shape:
    :param regularizer: if not `None`, use it to regularize weights
    :return:
    """
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def model_inference(input_tensor, regularizer):
    """
    Construct model and return the logits of output layer
    """
    if not HIDDEN_LAYER_NODE:
        with tf.variable_scope('out_layer'):
            weights = get_weight_variable([INPUT_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        return layer
    with tf.variable_scope('inpuy_layer'):
        weights = get_weight_variable([INPUT_NODE, HIDDEN_LAYER_NODE[0]], regularizer)
        biases = tf.get_variable('biases', [HIDDEN_LAYER_NODE[0]], initializer=tf.constant_initializer(0.0))
        layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    for idx, node in enumerate(HIDDEN_LAYER_NODE):
        if idx == 0:
            continue
        else:
            with tf.variable_scope('hidden_layer{}'.format(idx + 1)):
                weights = get_weight_variable([HIDDEN_LAYER_NODE[idx-1], HIDDEN_LAYER_NODE[idx]], regularizer)
                biases = tf.get_variable('biases', [HIDDEN_LAYER_NODE[idx]], initializer=tf.constant_initializer(0.0))
                layer = tf.nn.relu(tf.matmul(layer, weights) + biases)
    with tf.variable_scope('out_layer'):
        weights = get_weight_variable([HIDDEN_LAYER_NODE[-1], OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer = tf.matmul(layer, weights) + biases

    return layer

