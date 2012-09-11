#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from functions import sigmoid


class FANN(object):
    """
    A Functional Artificial Neural Network.
    """
    def __init__(self, layers):
        self.input_size = layers[0].input_dim
        self.output_size = layers[-1].output_dim
        self.layers = layers

    def forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning only the outcome.
        """
        activation = X
        for layer in self.layers:
            activation = sigmoid(layer.forward_pass(theta, activation))
        return activation

    def full_forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning the full list of activations.
        """
        activations = [X]
        for layer in self.layers:
            activations.append(sigmoid(layer.forward_pass(theta, activations[-1])))
        return activations

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        activations = self.full_forward_pass(theta, X)
        Y = activations[-1]
        delta = (T - Y) * Y * (1 - Y) #sigmoid
        grad_theta = self.layers[-1].calculate_gradient(activations[-2], delta).reshape(-1)
        for i in range(len(self.layers) - 1, 0, -1):
            a = activations[i]
            layer = self.layers[i]
            delta = layer.pass_down(delta) * a * (1 - a) #sigmoid
            grad = layer.calculate_gradient(a, delta)
            grad_theta = np.hstack((grad_theta, grad.reshape(-1)))
        return grad_theta


