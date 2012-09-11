#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from functions import sigmoid
from helpers import add_bias
from neural_nets.connections import FullConnection, FullConnectionWithBias


class FANN(object):
    """
    A Functional Artificial Neural Network.
      * 1 layer
    """
    def __init__(self, layer):
        self.input_size = layer.input_dim
        self.output_size = layer.output_dim
        self.layer = layer

    def forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X.
        """
        return sigmoid(self.layer.forward_pass(theta, X))

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        deltas = (T - Y) * Y * (1 - Y)
        X = np.atleast_2d(X)
        grad = self.layer.calculate_gradient(X, deltas)
        return grad.reshape(-1), Y, deltas