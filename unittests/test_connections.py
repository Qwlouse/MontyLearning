#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import FullConnection, FullConnectionWithBias, RecurrentConnection, ForwardAndRecurrentConnection, ForwardAndRecurrentSigmoidConnection
from helpers import *
from neural_nets.functions import error_function
from scipy.optimize import approx_fprime

theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
X_nb = X[:,:-1] # no bias included
T = np.array([[1, 0, 2, 1, 1]]).T

def test_FullConnection_dimensions():
    fc = FullConnection(5, 7)
    assert_equal(fc.input_dim, 5)
    assert_equal(fc.output_dim, 7)
    assert_equal(fc.get_param_dim(), 5*7)

def test_FullConnection_forward_pass_single_samples():
    fc = FullConnection(4, 1)
    for x, t in zip(X, T):
        t = np.atleast_2d(t)
        assert_equal(fc.forward_pass(theta, x), t)

def test_FullConnection_forward_pass_multi_sample():
    fc = FullConnection(4, 1)
    assert_equal(fc.forward_pass(theta, X), T)

def test_FullConnectionWithBias_dimensions():
    fc = FullConnectionWithBias(5, 7)
    assert_equal(fc.input_dim, 5)
    assert_equal(fc.output_dim, 7)
    assert_equal(fc.get_param_dim(), 6*7)

def test_FullConnectionWithBias_forward_pass_single_samples():
    fc = FullConnectionWithBias(3, 1)
    for x, t in zip(X_nb, T):
        t = np.atleast_2d(t)
        assert_equal(fc.forward_pass(theta, x), t)

def test_FullConnectionWithBias_forward_pass_multi_sample():
    fc = FullConnectionWithBias(3, 1)
    assert_equal(fc.forward_pass(theta, X_nb), T)

def test_RecurrentConnection_dimensions():
    rc = RecurrentConnection(5)
    assert_equal(rc.input_dim, 5)
    assert_equal(rc.output_dim, 5)
    assert_equal(rc.get_param_dim(), 5**2)

def test_RecurrentConnection_forward_pass_single_samples():
    rc = RecurrentConnection(4)
    theta = 2 * np.eye(4).flatten()
    for x in X:
        x = np.atleast_2d(x)
        # for single samples recurrence should not do anything
        assert_equal(rc.forward_pass(theta, x), x)

def test_RecurrentConnection_forward_pass_multi_sample():
    fc = RecurrentConnection(4)
    theta = np.eye(4).flatten()
    X_summed = np.array([[0, 0, 0, 1], [1, 0, 0, 2],[1, 1, 0, 3],[1, 1, 1, 4],[2, 2, 1, 5]])
    assert_equal(fc.forward_pass(theta, X), X_summed)

def test_ForwardAndRecurrentConnections_dimensions():
    frc = ForwardAndRecurrentConnection(3, 7)
    assert_equal(frc.input_dim, 3)
    assert_equal(frc.output_dim, 7)

def test_ForwardAndRecurrentConnections_param_dim():
    frc = ForwardAndRecurrentConnection(3, 7)
    assert_equal(frc.get_param_dim(), 3*7 + 7*7)

def test_ForwardAndRecurrentConnections_feed_forward_single_sample():
    # single sample, the recurrent connection should not jump in
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    assert_equal(frc.forward_pass(theta, 1), 1)

def test_ForwardAndRecurrentConnections_feed_forward_two_samples():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    X = np.array([[1],[1]])
    T = np.array([[1],[2]])
    assert_equal(frc.forward_pass(theta, X), T)

def test_ForwardAndRecurrentConnections_feed_forward_two_samples_using_carry():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    X = np.array([[1],[1]])
    T = np.array([[1],[2]])
    assert_equal(frc.forward_pass(theta, X[0]), T[0:1])
    assert_equal(frc.forward_pass(theta, X[1], X[0]), T[1:2])

def test_ForwardAndRecurrentConnections_backprop_single_sample():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    error, grad = frc.backprop(theta, 1, 1, 1)
    assert_equal(grad, [-1, 0])
    assert_equal(error, 1)

def test_ForwardAndRecurrentConnections_backprop_single_samples_with_carry():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    error, grad = frc.backprop(theta, 1, 1, 1, 1)
    assert_equal(grad, [-2, -1])
    assert_equal(error, 2)

def test_ForwardAndRecurrentConnections_backprop_two_samples():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    error, grad = frc.backprop(theta, [[1], [1]], [[1], [2]], [[-1], [-2]])
    assert_equal(grad, [5, 2])
    assert_equal(error, [[-3], [-2]])

def test_ForwardAndRecurrentConnections_backprop_gradient_check():
    frc = ForwardAndRecurrentConnection(1, 1)
    theta = np.ones(frc.get_param_dim())
    X = [[1.], [1.]]
    Y = [[1.], [2.]]
    T = np.array([[0.], [0.]])
    out_error = [[-1.], [-2.]]
    error, grad = frc.backprop(theta, X, Y, out_error)
    f = lambda t : error_function(T - frc.forward_pass(t, X))
    assert_almost_equal(approx_fprime(theta, f, 1e-8), grad)

def test_ForwardAndRecurrentConnections_backprop_random_example_gradient_check():
    frc = ForwardAndRecurrentConnection(4, 3)
    theta = np.random.randn(frc.get_param_dim())
    X = np.random.randn(10, 4)
    Y = frc.forward_pass(theta, X)
    T = np.zeros((10, 3))
    out_error = (T - Y)
    error, grad_c = frc.backprop(theta, X, Y, out_error)
    f = lambda t : error_function(T - frc.forward_pass(t, X))
    grad_e = approx_fprime(theta, f, 1e-8)
    assert_allclose(grad_c, grad_e, rtol=1e-3, atol=1e-5)

def test_ForwardAndRecurrentSigmoidConnections_backprop_random_example_gradient_check():
    frc = ForwardAndRecurrentSigmoidConnection(4, 3)
    theta = np.random.randn(frc.get_param_dim())
    X = np.random.randn(10, 4)
    Y = frc.forward_pass(theta, X)
    T = np.zeros((10, 3))
    out_error = (T - Y)
    error, grad_c = frc.backprop(theta, X, Y, out_error)
    f = lambda t : error_function(T - frc.forward_pass(t, X))
    grad_e = approx_fprime(theta, f, 1e-8)
    assert_allclose(grad_c, grad_e, rtol=1e-3, atol=1e-5)