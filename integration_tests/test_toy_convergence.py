#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from datasets import load_and, generate_majority_vote
from neural_nets.fann import FANN
from unittests.helpers import assert_less

def test_FANN_converges_on_and_problem():
    nn = FANN(2, 1)
    and_ = load_and()
    theta = np.array([-0.1, 0.1])
    for i in range(100):
        g, _, _ = nn.calculate_gradient(theta, and_.data, and_.target)
        theta -= g * 1
    error = nn.calculate_error(theta, and_.data, and_.target)
    assert_less(error,  0.2)

def test_FANN_converges_on_xor_problem():
    nn = FANN(2, 1)
    xor = load_and()
    theta = np.array([-0.1, 0.1])
    for i in range(100):
        g, _, _ = nn.calculate_gradient(theta, xor.data, xor.target)
        theta -= g * 1
    error = nn.calculate_error(theta, xor.data, xor.target)
    assert_less(error,  0.2)

def test_FANN_converges_on_vote_problem():
    nn = FANN(9, 1, include_bias=True)
    vote = generate_majority_vote()
    theta = np.zeros((10,))
    for i in range(500):
        g, _, _ = nn.calculate_gradient(theta, vote.data, vote.target)
        theta -= g * 1
    error = nn.calculate_error(theta, vote.data, vote.target)
    assert_less(error,  0.2)