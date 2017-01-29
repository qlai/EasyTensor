from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import utils

from est_base import EstBase


class MultiLayerPerceptron(EstBase):
    def __init__(self):
        ''' multilayer perceptron class for simple models
        if dropout == True: feed must include drop out probability named 'keep_prob', else feed includes 'input_data', 'target_data'
        hidden_layers and activations are lists of layer dimensions (int) and strings
        note train_step'''
        # TODO: add different cost functions

        # costfunc default to be cross_entropy
        super(MultiLayerPerceptron, self).__init__(784, 10, \
                                                   utils.cross_entropy, 0.3, \
                                                   'ADAM')

        self.hidden_dims = [2, 2, 2]
        self.activations = [u'relu', u'relu', u'relu']
        self.dropout = False

        #define placeholders for data
        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, self.input_dim], name='input_data')
            self.target_data = tf.placeholder(tf.float32, [None, self.output_dim], name='target_data')

        #define neural network
        self.num_layers = len(self.hidden_dims)

         
        hidden_layer_1 = utils.perceptron(self.input_data, self.input_dim, self.hidden_dims[0], \
                                'hidden_{}'.format(0), self.activations[0])
          
        hidden_layer_2 = utils.perceptron(hidden_layer_1, self.hidden_dims[0], \
                                                      self.hidden_dims[1], 'hidden_{}'.format(2), self.activations[1])
          
        hidden_layer_last = utils.perceptron(hidden_layer_2, self.hidden_dims[1], \
                                               self.hidden_dims[2], 'hidden_{}'.format(3), self.activations[2])
         

        # for i in range(self.num_layers):
        # 	if i == 0:
        # 		self.hidden.append(layer(self.input_data, self.input_dim, self.hidden_layers_dims[i], \
        # 						'hidden_{}'.format(i+1), self.activations[i]))
        #
        # 	# hidden_next = layer(self.hidden_previous, hidden_layers[i-1], hidden_layers[i], 'hidden_#number', activations[i])
        # 	else:
        # 		self.hidden.append(layer(hidden[i-1], self.hidden_layers_dims[i - 1], self.hidden_layers_dims[i], \
        # 						'hidden_{}'.format(i+1), self.activations[i]))

        if self.dropout:
            self.keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            dropped = tf.nn.dropout(hidden_layer_last, self.keep_prob)
            self.output = utils.perceptron(dropped, self.hidden_dims[-1], self.output_dim, 'output_layer')

        else:
            self.output = utils.perceptron(hidden_layer_last, self.hidden_dims[-1], self.output_dim, 'output_layer')

        #def cost with function, real y and output y
        with tf.name_scope('cost'):
            self.diff = self.costfunc(self.target_data, self.output)
            with tf.name_scope('total'):
                self.cost = tf.reduce_mean(self.diff)
                tf.summary.scalar('cost', self.cost)

        with tf.name_scope('train'):
            # self.train_step = utils.get_optimizer(self.learning_rate, self.cost, self.optimizer)
            if self.optimizer == 'ADAM':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            elif self.optimizer == 'GD':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            else:# default choice of optimizer to be GD
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)


        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target_data, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

        #define training operation