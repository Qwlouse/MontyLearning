#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import FullConnection, FullConnectionWithBias
from helpers import *

theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
X_nb = X[:,:-1] # no bias included
T = np.array([[1, 0, 2, 1, 1]]).T

def test_FullConnection_dimensions():
    fc = FullConnection(5, 7)
    assert_equal(fc.input_dim, 5)
    assert_equal(fc.output_dim, 7)

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

def test_FullConnectionWithBias_forward_pass_single_samples():
    fc = FullConnectionWithBias(3, 1)
    for x, t in zip(X_nb, T):
        t = np.atleast_2d(t)
        assert_equal(fc.forward_pass(theta, x), t)

def test_FullConnectionWithBias_forward_pass_multi_sample():
    fc = FullConnectionWithBias(3, 1)
    assert_equal(fc.forward_pass(theta, X_nb), T)