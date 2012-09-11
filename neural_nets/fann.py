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
        # NOTE: This method might be optimized
        _, activations = self.full_forward_pass(theta, X)
        return activations[-1]

    def full_forward_pass(self, theta, X):
        """
        Using the parameters theta as the weights evaluate this feed-forward
        neural network on the data X returning the sliced theta and full list of activations.
        """
        activations = [X]
        sliced_theta = self.slice_theta(theta)
        for layer, th in zip(self.layers,sliced_theta):
            activations.append(sigmoid(layer.forward_pass(th, activations[-1])))
        return sliced_theta, activations

    def calculate_error(self, theta, X, T):
        Y = self.forward_pass(theta, X)
        return np.sum(0.5 * (T - Y)**2)

    def calculate_gradient(self, theta, X, T):
        sliced_theta, activations = self.full_forward_pass(theta, X)
        Y = activations[-1]
        delta = (T - Y) * Y * (1 - Y) #sigmoid
        grad_theta = self.layers[-1].calculate_gradient(activations[-2], delta).reshape(-1)
        for i in range(len(self.layers) - 1, 0, -1):
            a = activations[i]
            layer = self.layers[i]
            delta = layer.backward_pass(sliced_theta[i], delta) * a * (1 - a) #sigmoid
            grad = self.layers[i-1].calculate_gradient(activations[i-1], delta)
            grad_theta = np.hstack((grad.reshape(-1), grad_theta))
        return grad_theta


