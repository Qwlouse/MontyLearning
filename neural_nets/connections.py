#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.helpers import add_bias


class FullConnection(object):
    """
    Simple linear feed-forward full connection without bias.
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def unpackTheta(self, theta):
        return theta.reshape(self.input_dim, self.output_dim)

    def forward_pass(self, theta, X):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        return X.dot(W)

    def backward_pass(self, theta, Y):
        W = self.unpackTheta(theta)
        return Y.dot(W.T)

    def calculate_gradient(self, X, delta):
        return -np.outer(delta, X)
