#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from datasets import load_and, generate_majority_vote, load_xor, generate_remember_pattern_over_time
from datasets.tools import seqEnum
from neural_nets.connections import FullConnectionWithBias, FullConnection, SigmoidLayer, ForwardAndRecurrentSigmoidConnection
from neural_nets.fann import FANN
from unittests.helpers import assert_less

def test_FANN_converges_on_and_problem():
    fc = FullConnection(2, 1)
    sig = SigmoidLayer(1)
    nn = FANN([fc, sig])
    and_ = load_and()
    theta = np.array([-0.1, 0.1])
    for i in range(100):
        g = nn.calculate_gradient(theta, and_.data, and_.target)
        theta -= g * 1
    error = nn.calculate_error(theta, and_.data, and_.target)
    assert_less(error,  0.2)

def test_FANN_converges_on_xor_problem():
    fc0 = FullConnectionWithBias(2, 2)
    fc1 = FullConnectionWithBias(2, 1)
    sig0 = SigmoidLayer(2)
    sig1 = SigmoidLayer(1)
    nn = FANN([fc0, sig0, fc1, sig1])
    xor = load_xor()
    theta = np.random.randn(nn.get_param_dim())
    for i in range(2000):
        g = nn.calculate_gradient(theta, xor.data, xor.target)
        theta -= g * 1
    error = nn.calculate_error(theta, xor.data, xor.target)
    assert_less(error,  0.4)

def test_FANN_converges_on_vote_problem():
    fc = FullConnectionWithBias(9, 1)
    sig = SigmoidLayer(1)
    nn = FANN([fc, sig])
    vote = generate_majority_vote()
    theta = np.zeros((10,))
    for i in range(500):
        g = nn.calculate_gradient(theta, vote.data, vote.target)
        theta -= g * 1
    error = nn.calculate_error(theta, vote.data, vote.target)
    assert_less(error,  0.2)

def test_RANN_converges_on_ropot_problem():
    frc = ForwardAndRecurrentSigmoidConnection(5, 5)
    nn = FANN([frc])
    rpot = generate_remember_pattern_over_time()
    theta = np.random.randn(nn.get_param_dim())
    for i in range(100):
        grad = np.zeros_like(theta)
        for X, T in seqEnum(rpot):
            grad += nn.calculate_gradient(theta, X, T)
        theta -= grad * 1

    error = np.sum(nn.calculate_error(theta, X, T) for X, T in seqEnum(rpot))
    assert_less(error, 10.)
