#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import add_bias, FullConnection, FullConnectionWithBias
from helpers import *


def test_add_bias_single_sample():
    x = np.array([2, 3, 4])
    x_wb = add_bias(x)
    assert_equal(x_wb.shape, (1, 4))
    assert_equal(x_wb, [[2, 3, 4, 1]])

def test_add_bias_many_samples():
    x = np.array([[2, 3, 4]]*5)
    x_wb = add_bias(x)
    assert_equal(x_wb.shape, (5, 4))
    assert_equal(x_wb, [[2, 3, 4, 1]]*5)

def test_FullConnection_dimensions():
    fc = FullConnection(5, 7)
    assert_equal(fc.input_size(), 5)
    assert_equal(fc.output_size(), 7)

def test_FullConnectionWithBias_dimensions():
    fc = FullConnectionWithBias(5, 7)
    assert_equal(fc.input_size(), 5)
    assert_equal(fc.output_size(), 7)

def test_FullConnection_pass_forward_single_samples():
    nn = FullConnection(4, 1)
    nn.weights = np.array([[-1, 1, 0, 1]]).T
    assert_equal(nn.pass_forward([0, 0, 0, 1]), 1)
    assert_equal(nn.pass_forward([1, 0, 0, 1]), 0)
    assert_equal(nn.pass_forward([0, 1, 0, 1]), 2)
    assert_equal(nn.pass_forward([0, 0, 1, 1]), 1)
    assert_equal(nn.pass_forward([1, 1, 0, 1]), 1)

def test_FullConnection_pass_forward_single_samples():
    nn = FullConnectionWithBias(3, 1)
    nn.weights = np.array([[-1, 1, 0, 1]]).T
    assert_equal(nn.pass_forward([0, 0, 0]), 1)
    assert_equal(nn.pass_forward([1, 0, 0]), 0)
    assert_equal(nn.pass_forward([0, 1, 0]), 2)
    assert_equal(nn.pass_forward([0, 0, 1]), 1)
    assert_equal(nn.pass_forward([1, 1, 0]), 1)