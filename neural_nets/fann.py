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
        self.theta_slices = self.get_theta_slices()

    def get_param_dim(self):
        dim = 0
        for l in self.layers:
            dim += l.get_param_dim()
        return dim

    def get_theta_slices(self):
        slices = {}
        theta_offset = 0
        for layer in self.layers:
            param_dim = layer.get_param_dim()
            slices[layer] = slice(theta_offset, theta_offset + param_dim)
            theta_offset += param_dim
        return slices

    def forward_pass(self, theta, X):
        # NOTE: This method might be optimized
        return self.full_forward_pass(theta, X)[-1]

    def full_forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning the full list of activations.
        """
        activations = [X]
        for layer in self.layers:
            activations.append(layer.forward_pass(theta[self.theta_slices[layer]], activations[-1]))
        return activations

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        activations = self.full_forward_pass(theta, X)
        Y = activations[-1]
        delta = (T - Y) * Y * (1 - Y) #sigmoid
        grad_theta = np.array([])
        for i in range(len(self.layers) - 1, -1, -1):
            a = activations[i]
            layer = self.layers[i]
            th = theta[self.theta_slices[layer]]
            grad = self.layers[i].calculate_gradient(a, delta)
            grad_theta = np.hstack((grad.reshape(-1), grad_theta))
            if i > 0 : # skip the last delta
                delta = layer.backward_pass(th, delta) * a * (1 - a) #sigmoid
        return grad_theta


