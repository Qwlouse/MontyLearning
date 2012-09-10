#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from helpers import *

from neural_nets.fann import FANN

theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
T = np.array([[1, 0, 2, 1, 1]]).T


def test_FANN_dimensions():
    nn = FANN(5, 1)
    assert_equal(nn.input_size, 5)
    assert_equal(nn.output_size, 1)

def test_FANN_feed_forward_single_sample():
    nn = FANN(4, 1)
    for x, t in zip(X, T) :
        assert_equal(nn.forward_pass(theta, x), t)

def test_FANN_feed_forward_multisample():
    nn = FANN(4, 1)
    assert_equal(nn.forward_pass(theta, X), T)

