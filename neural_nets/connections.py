#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np


class ConnectionLayer(object):
    """
    Base class for all connections.
    """
    @property
    def input_size(self):
        raise NotImplementedError()

    @property
    def output_size(self):
        raise NotImplementedError()

    def is_recurrent(self):
        return False


class FullConnection(ConnectionLayer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)

    def input_size(self):
        return self.weights.shape[1]

    def output_size(self):
        return self.weights.shape[0]

    def pass_forward(self, X):
        return self.weights.dot(X)

    def pass_backward(self, Y):
        return self.weights.T.dot(Y)

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


def add_bias(x):
    return np.hstack((x, [1.]))

class FullConnectionWithBias(FullConnection):
    def __init__(self, input_size, output_size):
        super(FullConnectionWithBias, self).__init__(input_size + 1, output_size)

    def pass_forward(self, X):
        return FullConnection.pass_forward(self, add_bias(X))

    def pass_backward(self, Y):
        return FullConnection.pass_backward(self, Y)[:-1]

    def calculate_gradient(self, X, delta):
        return FullConnection.calculate_gradient(self, add_bias(X), delta)