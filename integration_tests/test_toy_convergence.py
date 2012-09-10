#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from datasets import load_and
from neural_nets.fann import FANN

def test_FANN_converges_on_and_problem():
    nn = FANN(2, 1)
    and_ = load_and()
    theta = np.array([-0.1, 0.1])
    for i in range(100):
        g, _, _ = nn.calculate_gradient(theta, and_.data, and_.target)
        theta -= g * 1
    error = nn.calculate_error(theta, and_.data, and_.target)
    assert error < 0.2