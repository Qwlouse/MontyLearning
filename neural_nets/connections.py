#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np


class ConnectionLayer(object):
    """
    Base class for all connections.
    """
    def input_size(self):
        raise NotImplementedError()

    def output_size(self):
        raise NotImplementedError()

    def is_recurrent(self):
        return False


class FullConnection(ConnectionLayer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)

    def input_size(self):
        return self.weights.shape[0]

    def output_size(self):
        return self.weights.shape[1]

    def pass_forward(self, X):
        return X.dot(self.weights)

    def pass_backward(self, Y):
        return Y.dot(self.weights.T)

    def calculate_gradient(self, X, delta):
        return -np.outer(delta, X)

    def gradient_estimation_process(self, epsilon):
        for r in range(self.weights.shape[0]):
            for c in range(self.weights.shape[1]):
                w = self.weights[r, c]
                self.weights[r, c] = w + epsilon
                yield r, c
                self.weights[r, c] = w - epsilon
                yield
                self.weights[r,c] = w


def add_bias(X):
    X = np.atleast_2d(X)
    return np.hstack((X, np.ones((X.shape[0], 1))))

class FullConnectionWithBias(FullConnection):
    def __init__(self, input_size, output_size):
        super(FullConnectionWithBias, self).__init__(input_size + 1, output_size)

    def input_size(self):
        return FullConnection.input_size(self) - 1

    def pass_forward(self, X):
        return FullConnection.pass_forward(self, add_bias(X))

    def pass_backward(self, Y):
        return FullConnection.pass_backward(self, Y)[:-1]

    def calculate_gradient(self, X, delta):
        return FullConnection.calculate_gradient(self, add_bias(X), delta)