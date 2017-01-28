from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import utils

from est_base import EstBase


class Recurrent_NN(EstBase):
    def __init__(self, input_dim, output_dim, hidden_dims, \
                 activations, learning_rate, dropout = False, \
                 costfunc = utils.cross_entropy, optimizer='ADAM'):


        super(Recurrent_NN, self).__init__(input_dim, output_dim, \
                                                   costfunc, learning_rate, optimizer)
