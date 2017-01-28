from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import *
import train as op
import multilayer_perceptron as base
import sys


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    # loading data object
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    model = base.MultiLayerPerceptron(input_dim=784, output_dim=10, \
                                      hidden_layers=[256,256], activations=['relu', None], \
                                      learning_rate=0.7, dropout = False, \
                                      costfunc = cross_entropy)

    log_dir = '/Users/AndyZhang/Cambridge/hack/bins'
    op.train(model, dataset=mnist, NUM_ITERS=100, BATCH_SIZE=100,\
             LOG_DIR=log_dir, KEEP_PROB=1.)
    op.test(model, dataset=mnist, LOG_DIR=log_dir, TEST_ID = 0)






