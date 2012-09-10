#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import add_bias
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