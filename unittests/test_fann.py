#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from scipy.optimize import approx_fprime
from helpers import *
from neural_nets.connections import FullConnection, FullConnectionWithBias

from neural_nets.fann import FANN
from neural_nets.functions import sigmoid

theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
X_nb = X[:,:-1] # no bias included
T = sigmoid(np.array([[1, 0, 2, 1, 1]]).T)
E = 0.5 * T**2

def test_FANN_dimensions():
    fc = FullConnection(5, 1)
    nn = FANN([fc])
    assert_equal(nn.input_size, 5)
    assert_equal(nn.output_size, 1)

def test_FANN_with_bias_dimensions():
    fc = FullConnectionWithBias(5, 1)
    nn = FANN([fc])
    assert_equal(nn.input_size, 5)
    assert_equal(nn.output_size, 1)

def test_FANN_feed_forward_single_sample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    for x, t in zip(X, T) :
        t = np.atleast_2d(t)
        assert_equal(nn.forward_pass(theta, x), t)

def test_FANN_with_bias_feed_forward_single_sample():
    fc = FullConnectionWithBias(3, 1, function=sigmoid)
    nn = FANN([fc])
    for x, t in zip(X_nb, T) :
        t = np.atleast_2d(t)
        assert_equal(nn.forward_pass(theta, x), t)

def test_FANN_error_single_sample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    for x, t, e in zip(X, T, E) :
        assert_equal(nn.calculate_error(theta, x, t), 0)
        assert_equal(nn.calculate_error(theta, x, 0), e)

def test_FANN_feed_forward_multisample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    assert_equal(nn.forward_pass(theta, X), T)

def test_FANN_with_bias_feed_forward_multisample():
    fc = FullConnectionWithBias(3, 1, function=sigmoid)
    nn = FANN([fc])
    assert_equal(nn.forward_pass(theta, X_nb), T)

def test_FANN_error_multisample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    assert_equal(nn.calculate_error(theta, X, T), 0.0)
    assert_equal(nn.calculate_error(theta, X, np.zeros_like(T)), np.sum(E))

def test_FANN_with_bias_error_multisample():
    fc = FullConnectionWithBias(3, 1, function=sigmoid)
    nn = FANN([fc])
    assert_equal(nn.calculate_error(theta, X_nb, T), 0.0)
    assert_equal(nn.calculate_error(theta, X_nb, np.zeros_like(T)), np.sum(E))

def test_FANN_gradient_single_sample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    theta = np.random.randn(nn.get_param_dim())
    for x, t in zip(X, T) :
        grad_c = nn.calculate_gradient(theta, x, t)
        grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, x, t)
        assert_almost_equal(grad_c, grad_e)

def test_FANN_with_bias_gradient_single_sample():
    fc = FullConnectionWithBias(3, 1, function=sigmoid)
    nn = FANN([fc])
    theta = np.random.randn(nn.get_param_dim())
    for x, t in zip(X_nb, T) :
        grad_c = nn.calculate_gradient(theta, x, t)
        grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, x, t)
        assert_almost_equal(grad_c, grad_e)

def test_FANN_gradient_multisample():
    fc = FullConnection(4, 1, function=sigmoid)
    nn = FANN([fc])
    theta = np.random.randn(nn.get_param_dim())
    grad_c = nn.calculate_gradient(theta, X, T)
    grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, X, T)
    assert_almost_equal(grad_c, grad_e)

def test_FANN_with_bias_gradient_multisample():
    fc = FullConnectionWithBias(3, 1, function=sigmoid)
    nn = FANN([fc])
    theta = np.random.randn(nn.get_param_dim())
    grad_c = nn.calculate_gradient(theta, X_nb, T)
    grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, X_nb, T)
    assert_almost_equal(grad_c, grad_e)

def test_FANN_multilayer_gradient_single_sample():
    fc0 = FullConnection(4, 2, function=sigmoid)
    fc1 = FullConnection(2, 1, function=sigmoid)
    nn = FANN([fc0, fc1])
    theta = np.random.randn(nn.get_param_dim())
    for x, t in zip(X, T) :
        grad_c = nn.calculate_gradient(theta, x, t)
        grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, x, t)
        assert_almost_equal(grad_c, grad_e)

def test_FANN_with_bias_multilayer_gradient_single_sample():
    fc0 = FullConnectionWithBias(3, 2, function=sigmoid)
    fc1 = FullConnectionWithBias(2, 1, function=sigmoid)
    nn = FANN([fc0, fc1])
    theta = np.random.randn(nn.get_param_dim())
    for x, t in zip(X_nb, T) :
        grad_c = nn.calculate_gradient(theta, x, t)
        grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, x, t)
        assert_almost_equal(grad_c, grad_e)

def test_FANN_multilayer_gradient_multisample():
    fc0 = FullConnectionWithBias(4, 2, function=sigmoid)
    fc1 = FullConnectionWithBias(2, 1, function=sigmoid)
    nn = FANN([fc0, fc1])
    theta = np.random.randn(nn.get_param_dim())
    grad_c = nn.calculate_gradient(theta, X, T)
    grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, X, T)
    assert_almost_equal(grad_c, grad_e)

def test_FANN_multilayer_with_bias_gradient_multisample():
    fc0 = FullConnectionWithBias(3, 2, function=sigmoid)
    fc1 = FullConnectionWithBias(2, 1, function=sigmoid)
    nn = FANN([fc0, fc1])
    theta = np.random.randn(nn.get_param_dim())
    grad_c = nn.calculate_gradient(theta, X_nb, T)
    grad_e = approx_fprime(theta, nn.calculate_error, 1e-8, X_nb, T)
    assert_almost_equal(grad_c, grad_e)