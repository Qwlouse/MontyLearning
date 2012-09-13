#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.functions import identity, sigmoid
from neural_nets.helpers import add_bias


class FullConnection(object):
    """
    Feed-forward full connection without bias.
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        return X.dot(W)

    def backprop(self, theta, X, Y, out_error):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        grad = -X.T.dot(out_error)
        in_error = out_error.dot(W.T)
        return in_error, grad

class FullConnectionWithBias(FullConnection):
    """
    Feed-forward full connection WITH bias.
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

class RecurrentConnection(object):
    """
    Recurrent full connection without bias.
    """
    def __init__(self, dim):
        self.input_dim = dim
        self.output_dim = dim

    def get_param_dim(self):
        """
        Return the dimension of the parameter-space.
        """
        return self.input_dim * self.output_dim

    def unpackTheta(self, theta):
        return theta.reshape(self.input_dim, self.output_dim)

    def forward_pass(self, theta, X, carry = None):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        if carry is None:
            carry = np.zeros_like(X[0])
        Y = np.zeros_like(X)
        for i, x in enumerate(X):
            Y[i] = (x + carry)
            carry = Y[i].dot(W)
        return Y

    def backprop(self, theta, X, Y, out_error):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        carry = np.zeros_like(out_error[0])
        grad = np.zeros_like(W)
        in_error = np.zeros_like(out_error)
        for i in range(len(out_error)-1, -1, -1):
            delta = out_error[i] + carry.dot(W.T)
            x = X[i:i+1] # slice indexing to preserve 2d
            grad -= x.T.dot(delta)
            in_error[i] = delta
            carry =  delta
        return in_error, grad


class ForwardAndRecurrentConnection(object):
    """
    Set of connections that connect two layers and also recurrently connects the output layer
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_param_dim(self):
        """
        Return the dimension of the parameter-space.
        """
        return self.input_dim * self.output_dim + self.output_dim ** 2

    def unpackTheta(self, theta):
        W_in_dim = self.input_dim * self.output_dim
        W_in = theta[:W_in_dim].reshape(self.input_dim, self.output_dim)
        W_r  = theta[W_in_dim:].reshape(self.output_dim, self.output_dim)
        return W_in, W_r

    def forward_pass(self, theta, X, carry=None):
        W_in, W_r = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        if carry is None:
            carry = np.zeros_like(X[0])
        Y = np.zeros_like(X)
        for i, x in enumerate(X):
            carry = x.dot(W_in) + carry.dot(W_r)
            Y[i] = carry
        return Y

    def backprop(self, theta, X, Y, out_error):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        grad = -X.T.dot(out_error)
        in_error = out_error.dot(W.T)
        return in_error, grad



class SigmoidLayer(object):
    def __init__(self, dim):
        self.input_dim = dim
        self.output_dim = dim

    def get_param_dim(self):
        return 0

    def forward_pass(self, theta, X):
        return sigmoid(X)

    def backprop(self, theta, X, Y, out_error):
        return out_error * Y * (1-Y), np.array([])