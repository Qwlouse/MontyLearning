#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import add_bias, FullConnection, FullConnectionWithBias
from helpers import *


def test_add_bias_single_sample():
    x = np.array([2, 3, 4])
    x_wb = add_bias(x)
    assert_equal(x_wb.shape, (4,))
    assert_equal(x_wb, [2, 3, 4, 1])

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

