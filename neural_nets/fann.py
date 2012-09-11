#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from functions import sigmoid
from helpers import add_bias
from neural_nets.connections import FullConnection


class FANN(object):
    """
    A Functional Artificial Neural Network.
      * 1 layer
    """
    def __init__(self, input_size, output_size, include_bias=False):
        self.input_size = input_size
        self.output_size = output_size
        self.layer = FullConnection(input_size, output_size)
        self.include_bias = include_bias

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
        if self.include_bias:
            X = add_bias(X)
        grad = -X.T.dot(deltas)
        return grad.reshape(-1), Y, deltas