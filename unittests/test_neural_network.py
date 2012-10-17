#!/usr/bin/python
# coding=utf-8
"""
docstring
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.optimize import approx_fprime
from helpers import *
from neural_nets.connections import FullConnection, FullConnectionWithBias, SigmoidLayer, RecurrentConnection, ForwardAndRecurrentConnection, ForwardAndRecurrentSigmoidConnection

from neural_nets.neural_network import NeuralNetwork
from neural_nets.functions import sigmoid

theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
X_nb = X[:,:-1] # no bias included
T = sigmoid(np.array([[1, 0, 2, 1, 1]]).T)
E = 0.5 * T**2

def test_FANN_dimensions():
    fc = FullConnection(5, 1)
    nn = NeuralNetwork([(fc, 0, 0)])
    assert_equal(nn.input_size, 5)
    assert_equal(nn.output_size, 1)

def test_FANN_with_bias_dimensions():
    fc = FullConnectionWithBias(5, 1)
    nn = NeuralNetwork([(fc, 0, 0)])
    assert_equal(nn.input_size, 5)
    assert_equal(nn.output_size, 1)

def test_FANN_feed_forward_single_sample():
    fc = FullConnection(4, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    for x, t in zip(X, T) :
        t = np.atleast_2d(t)
        x = np.atleast_2d(x)
        assert_equal(nn.forward_pass(theta, x), t)

def test_FANN_with_bias_feed_forward_single_sample():
    fc = FullConnectionWithBias(3, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    for x, t in zip(X_nb, T) :
        t = np.atleast_2d(t)
        x = np.atleast_2d(x)
        assert_equal(nn.forward_pass(theta, x), t)

def test_FANN_error_single_sample():
    fc = FullConnection(4, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    for x, t, e in zip(X, T, E) :
        x = np.atleast_2d(x)
        t = np.atleast_2d(t)
        assert_equal(nn.calculate_error(theta, x, t), 0)
        assert_equal(nn.calculate_error(theta, x, 0), e)

def test_FANN_feed_forward_multisample():
    fc = FullConnection(4, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    assert_equal(nn.forward_pass(theta, X), T)

def test_FANN_with_bias_feed_forward_multisample():
    fc = FullConnectionWithBias(3, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    assert_equal(nn.forward_pass(theta, X_nb), T)

def test_FANN_error_multisample():
    fc = FullConnection(4, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    assert_equal(nn.calculate_error(theta, X, T), 0.0)
    assert_equal(nn.calculate_error(theta, X, np.zeros_like(T)), np.sum(E))

def test_FANN_with_bias_error_multisample():
    fc = FullConnectionWithBias(3, 1)
    sig = SigmoidLayer(1)
    nn = NeuralNetwork([(fc, 0, 0), (sig, 0, 0)])
    assert_equal(nn.calculate_error(theta, X_nb, T), 0.0)
    assert_equal(nn.calculate_error(theta, X_nb, np.zeros_like(T)), np.sum(E))

