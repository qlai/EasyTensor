from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import warnings

import tensorflow as tf

import utils

from est_base import EstBase


class Recurrent_NN(EstBase):
    def __init__(self, input_dim, output_dim, \
    			 cell_type, cell_dim, num_layers, \
                 out_activation,\
                 costfunc = utils.cross_entropy, learning_rate = 0.5, optimizer='ADAM'):
    # (28, 10, 28, ['LSTM', 'LSTM'], [128, 128], 'softmax')
    # (28, 10, 28, ['LSTM', 'GRU'], [128, 128], 'softmax')
    '''RNN is v. different to perceptron etc. 
    	NOTE: cell_types, state_dims, steps, as new inputs, and only 1 str for out_activation
    '''

        super(Recurrent_NN, self).__init__(input_dim, output_dim, \
                                                   costfunc, learning_rate, optimizer)

        self.dropout = dropout

        self.steps = input_dim[0]
        self.chunk_size = input_dim[1]


        #define placeholders for data
        with tf.name_scope('input'):
            self.input_data = tf.placeholder(tf.float32, [None, self.input_dim[0]*self.input_dim[1]], name='input_data')
            self.target_data = tf.placeholder(tf.float32, [None, self.output_dim], name='target_data')

        #define neural network
        self.num_layers = len(hidden_dims)
        self.input_data_adj = tf.reshape(self.input_data, [-1, self.steps, self.chunk_size])
        self.cell_type = cell_type
        self.cell_dim = cell_dim

        with tf.name_scope("RNN"):

	        if self.cell_type == "BASIC":
	        	cell = tf.nn.rnn_cell.BasicRNNCell(cell_dim)
	        elif self.cell_type == "LSTM":
	        	cell = tf.nn.rnn_cell.BasicLSTMCell(cell_dim)
	        elif self.cell_type == "GRU":
	        	cell = tf.nn.rnn_cell.BasicGRU(cell_dim)
	        else:
	        	warnings.warn('{} is not implemented, using LSTM instead'.format(self.cell_type))
                cell = tf.nn.rnn_cell.BasicLSTMCell(cell_dim)

            if self.cell_type == 'BASIC' or self.cell_type == 'GRU':
                self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple = False)
            else:
                self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        rnn_outputs, final_state = tf.nn.rnn(self.cell, self.input_data_adj)


        with tf.variable_scope("output"):
            self.output = utils.perceptron(rnn_outputs[-1], self.state_dims[-1], self.output_dim, 'output_layer', act=out_activation)


        with tf.name_scope('cost'):
            self.diff = self.costfunc(self.target_data, self.output)
            with tf.name_scope('total'):
                self.cost= tf.reduce_mean(self.diff)
                tf.summary.scalar('cost', self.cost)
                
        with tf.name_scope('train'):
            # self.train_step = utils.get_optimizer(self.learning_rate, self.cost, self.optimizer)
            if optimizer=='ADAM':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            elif optimizer=='GD':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            else:# default choice of optimizer to be GD
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
