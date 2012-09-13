#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import FullConnection, FullConnectionWithBias, RecurrentConnection, ForwardAndRecurrentConnection
from helpers import *

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