#!/usr/bin/python
# coding=utf-8
"""
docstring
"""
from __future__ import division, print_function, unicode_literals
import numpy as np

class NeuralNetwork(object):
    """
    A general neural network in a functional style
    """
    def __init__(self, layers):
        self.input_size = layers[0][0].input_dim
        self.output_size = layers[-1][0].output_dim
        self.layers = layers
        self.theta_slices = self.get_theta_slices()

    def get_param_dim(self):
        dim = 0
        for l, t, a in self.layers:
            dim += l.get_param_dim()
        return dim

    def get_theta_slices(self):
        slices = {}
        theta_offset = 0
        for layer, t, a in self.layers:
            param_dim = layer.get_param_dim()
            slices[layer] = slice(theta_offset, theta_offset + param_dim)
            theta_offset += param_dim
        return slices

    def forward_pass(self, theta, X):
        # NOTE: This method might be optimized
        activations = self.full_forward_pass(theta, X)
        return np.array([activations[t][-1][0] for t in range(len(X))])

    def full_forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning the full list of activations.
        """
        activations = []
        for t, x in enumerate(X):
            activations.append([x])
            for a, (layer, t_offset, a_offset) in enumerate(self.layers):
                input = activations[t+t_offset][a + a_offset]
                th = theta[self.theta_slices[layer]]
                activations[t].append(layer.forward_pass(th, input))
        return activations

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        activations = self.full_forward_pass(theta, X)
        Y = activations[-1]
        delta = (T - Y)
        grad_theta = np.array([])
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            th = theta[self.theta_slices[layer]]
            input = activations[i]
            output = activations[i+1]
            delta, grad = layer.backprop(th, input, output, delta)
            grad_theta = np.hstack((grad.reshape(-1), grad_theta))

        return grad_theta