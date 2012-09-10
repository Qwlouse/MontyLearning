#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np

class FANN(object):
    """
    A Functional Artificial Neural Network.
      * 1 layer
      * linear output
      * no bias
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def unpack(self, theta):
        return theta.reshape(self.input_size, self.output_size)

    def forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X.
        """
        W = self.unpack(theta)
        return X.dot(W)

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        deltas = T - Y
        X = np.atleast_2d(X)
        grad = -X.T.dot(deltas)
        return grad.reshape(-1), Y, deltas