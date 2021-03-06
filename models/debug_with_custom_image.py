from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import utils
import train as op
import multilayer_perceptron as base
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Debugging Multilayer_perceptron
also providing a way of using the libraries
'''

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    # loading data object
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    model = base.MultiLayerPerceptron(input_dim=784, output_dim=10, \
                                      hidden_dims=[512], activations=['relu', None], \
                                      learning_rate=0.3, dropout = False, \
                                      costfunc = utils.cross_entropy, optimizer='GD')


    # model = base.MultiLayerPerceptron()


    # log_dir = '/Users/AndyZhang/Cambridge/hack/bins'
    log_dir = '.'
    save_path = log_dir+'/model_temp.ckpt'
    op.train(model, dataset=mnist, NUM_ITERS=200, BATCH_SIZE=100,\
             LOG_DIR=log_dir, KEEP_PROB=1., TEST=True, SAVE_PATH = save_path)

    test_data = cv2.imread('../test_data/7.png', cv2.IMREAD_GRAYSCALE)
    test_data = test_data / 255
    # test_data = mnist.test.images[0]
    # make a color map of fixed colors
    # cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
    # bounds = [-2, -1, 1, 2]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #
    # # tell imshow about color map so that only set colors are used
    # img = plt.imshow(test_data, interpolation='nearest',
    #                     cmap=cmap, norm=norm)
    #
    # # make a color bar
    # plt.colorbar(img, cmap=cmap,
    #                 norm=norm, boundaries=bounds, ticks=[-1, 0, 1])
    #
    # plt.show()
    test_data = test_data.flatten()
    test_data = 1 - test_data
    test_data = test_data[np.newaxis, :]
    # y = op.predict(model, save_path, mnist.test.images)
    y = op.predict(model, save_path, test_data)
    print(np.shape(y))
    print(y[0,:])
    print(np.argmax(y[0,:]))
    print(np.argmax(y[:5,:],1))






