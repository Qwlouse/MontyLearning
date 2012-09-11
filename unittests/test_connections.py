#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from neural_nets.connections import FullConnection
from helpers import *

def test_FullConnection_dimensions():
    fc = FullConnection(5, 7)
    assert_equal(fc.input_dim, 5)
    assert_equal(fc.output_dim, 7)

def test_FullConnection_pass_forward_single_samples():
    fc = FullConnection(4, 1)
    theta = np.array([-1, 1, 0, 1])
    assert_equal(fc.forward_pass(theta, [0, 0, 0, 1]), 1)
    assert_equal(fc.forward_pass(theta, [1, 0, 0, 1]), 0)
    assert_equal(fc.forward_pass(theta, [0, 1, 0, 1]), 2)
    assert_equal(fc.forward_pass(theta, [0, 0, 1, 1]), 1)
    assert_equal(fc.forward_pass(theta, [1, 1, 0, 1]), 1)
