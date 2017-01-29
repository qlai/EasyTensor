
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the pre-trained model .ckpt file')
    parser.add_argument('--image_path', type=str, help='path to the images to be tested')
    FLAGS, unparsed = parser.parse_known_args()
    # test
    test_images = cv2.imread(FLAGS.image_path, cv2.IMREAD_GRAYSCALE)
    test_images = test_images.flatten()
    test_images = test_images/255
    test_images = 1 - test_images
    test_images = test_images[np.newaxis, :]
    y = predict_by_modelMLP(FLAGS.model_path, test_images)
    result = np.argmax(y, 1)
    print(result)








