import tensorflow as tf

activation_funcs = {
    'relu' : tf.nn.relu,
    'relu6' : tf.nn.relu6,
    'crelu' : tf.nn.crelu,
    'elu' : tf.nn.elu,
    'softplus' : tf.nn.softplus,
    'softsign' : tf.nn.softsign,
    'sigmoid' : tf.sigmoid,
    'tanh' : tf.tanh

}

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def perceptron(input, input_dim, output_dim, layer_name, act = None, summaries = True):
    '''simple layer wrapper for perceptrons'''
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            weights = weight_variable([input_dim, output_dim])
            if summaries:
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            if summaries:
                variable_summaries(biases)
        with tf.name_scope('output'):
            if act == None:
                output = tf.matmul(input, weights) + biases
            else:
                output_ = tf.matmul(input, weights) + biases
                output = map(activation_funcs[act], [output_])[0]

    tf.summary.histogram('output', output)

    return output

def flatten():
    pass

def convolution_layer(input, input_dim, output_dim, patchsize, layer_name, act = 'relu', summaries = True):
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            weights = weight_variable([patchsize[0], patchsize[1], input_dim, output_dim])
            if summaries:
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            if summaries:
                variable_summaries(biases)
        with tf.name_scope('output'):
            if act == None:
                output = conv2d(input, weights)
                output = tf.nn.bias_add(output,biases)
                output = max_pool_2x2(output)
            else:
                output = conv2d(input, weights)
                output = tf.nn.bias_add(output,biases)
                output = map(activation_funcs[act], [output_])[0]
                output = max_pool_2x2(output)

    tf.summary.histogram('output', output)

    return output


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def cross_entropy(labels, logits):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return diff
