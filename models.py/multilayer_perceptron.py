from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from utils import *


class MultiLayerPerceptron():
	def __init__(input_dim, output_dim, hidden_layers, activations, learning_rate, dropout = False, costfunc = cross_entropy): 
	             #hidden layers and activations are lists of layer dimensions (int) and strings
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_layers = hidden_layers
		self.activations = activations
		self.costfunc = costfunc
		self.learning_rate = learning_rate

		#define placeholders for data
		with tf.name_scope('input'):
			self.input = tf.placeholder(tf.float64, [None, input_dim], name='input_data')
		    self.target = tf.placeholder(tf.float64, [None, output_dim], name='output_data')

		#define neural network
		self.num_layers = len(hidden_layers)

		self.hidden = []

		for i in range(self.num_layers):
			if i == 0:
				self.hidden.append(layer(self.input, input_dim, hidden_layers[i], \
								'hidden_{}'.format(i+1), activations[i]))
			else:
				self.hidden.append(layer(hidden[i-1], input_dim, hidden_layers[i], \
								'hidden_{}'.format(i+1), activations[i]))

		if dropout:
		    keep_prob = tf.placeholder(tf.float32)
		    tf.summary.scalar('dropout_keep_probability', keep_prob)
		    dropped = tf.nn.dropout(self.hidden[-1], keep_prob)
		    self.output = layer(dropped, self.hidden_layers[-1], self.output_dim, 'output_layer')

		else:
			self.output = layer(self.hidden[-1], self.hidden_layers[-1], self.output_dim, 'output_layer')

		#def cost with function, real y and output y 
		with tf.name_scope('cost'):
			self.diff = self.costfunc(self.target, self.output)
			with tf.name_scope('total'):
				self.cost= tf.reduce_mean(self.diff)
			    tf.summary.scalar('cost', self.cost)

		with tf.name_scope('accuracy'):
		    with tf.name_scope('correct_prediction'):
			    correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.target, 1))
		    with tf.name_scope('accuracy'):
		    	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)


		#define training operation
		with tf.name_scope('train'):
		    self.train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
		        self.cost)
