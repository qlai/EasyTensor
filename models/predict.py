
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import utils
import train as op
import multilayer_perceptron_output as base
import numpy as np


def predict_by_modelMLP(model_path, test_images):
    '''
    NOTE, this only works for MultiLayerPerceptron models
    :param model_path: path to the previously trained model, i.e. model.ckpt
    :param test_images:
    :return: predicted y
    '''
    model = base.MultiLayerPerceptron()
    # model = base.MultiLayerPerceptron(input_dim=784, output_dim=10, \
    #                                   hidden_dims=[512], activations=['relu', None], \
    #                                   learning_rate=0.3, dropout = False, \
    #                                   costfunc = utils.cross_entropy, optimizer='GD')

    y = op.predict(model, model_path, test_images)
    return y









