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

    def get_param_dim(self):
        dim = 0
        for l in self.layers:
            dim += l.get_param_dim()
        return dim

    def slice_theta(self, theta):
        slices = []
        theta_offset = 0
        for layer in self.layers:
            param_dim = layer.get_param_dim()
            slices.append(theta[theta_offset:theta_offset + param_dim])
            theta_offset += param_dim
        return slices

    def forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning only the outcome.
        """
        # NOTE: This method might be optimized
        return self.full_forward_pass(theta, X)[-1]

    def full_forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning the full list of activations.
        """
        activations = [X]
        for layer, th in zip(self.layers, self.slice_theta(theta)):
            activations.append(sigmoid(layer.forward_pass(th, activations[-1])))
        return activations

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        activations = self.full_forward_pass(theta, X)
        Y = activations[-1]
        delta = (T - Y) * Y * (1 - Y) #sigmoid
        grad_theta = self.layers[-1].calculate_gradient(activations[-2], delta).reshape(-1)
        theta_slices = self.slice_theta(theta)
        for i in range(len(self.layers) - 1, 0, -1):
            a = activations[i]
            layer = self.layers[i]
            delta = layer.backward_pass(theta_slices[i], delta) * a * (1 - a) #sigmoid
            grad = layer.calculate_gradient(a, delta)
            grad_theta = np.hstack((grad.reshape(-1), grad_theta))
        return grad_theta


