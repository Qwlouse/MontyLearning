#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.functions import identity
from neural_nets.helpers import add_bias


class FullConnection(object):
    """
    Simple feed-forward full connection without bias.
    """
    def __init__(self, input_dim, output_dim, function=identity):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f = function

    def get_param_dim(self):
        """
        Return the dimension of the parameter-space.
        """
        return self.input_dim * self.output_dim

    def unpackTheta(self, theta):
        return theta.reshape(self.input_dim, self.output_dim)

    def forward_pass(self, theta, X):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        return self.f(X.dot(W))

    def backprop(self, theta, X, Y, out_error):
        delta = out_error * self.f.reverse(Y)
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        grad = -X.T.dot(delta)
        in_error = delta.dot(W.T)
        return in_error, grad

class FullConnectionWithBias(FullConnection):
    """
    Linear feed-forward full connection WITH bias.
    """
    def get_param_dim(self):
        """
        Return the dimension of the parameter-space.
        """
        return (self.input_dim + 1) * self.output_dim

    def unpackTheta(self, theta):
        return theta.reshape(self.input_dim + 1, self.output_dim)

    def forward_pass(self, theta, X):
        return FullConnection.forward_pass(self, theta, add_bias(X))

    def backprop(self, theta, X, Y, out_error):
        in_error, grad = FullConnection.backprop(self, theta, add_bias(X), Y, out_error)
        return in_error[:,:-1], grad