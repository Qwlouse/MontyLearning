#!/usr/bin/python
# coding=utf-8
"""
# This is a demo experiment to explore the workflow.
# I'll (mis)use this docstring to give a description and also
# contain the configuration file.
seed = 1234567
learning_rate = 0.01
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from experiment import Experiment
from neural_nets.connections import FullConnectionWithBias, SigmoidLayer
from neural_nets.fann import FANN

ex = Experiment(__doc__)

@ex.stage
def center(X):
    return X - np.mean(X, 0)

@ex.stage
def build_nn():
    l0 = FullConnectionWithBias(10, 10)
    s0 = SigmoidLayer(10)
    l1 = FullConnectionWithBias(10, 2)
    s1 = SigmoidLayer(2)
    return FANN([l0, s0, l1, s1])

@ex.stage
def initialize(nn, rnd):
    return rnd.randn(nn.get_param_dim())

@ex.main
def main():
    # load data
    X = np.zeros(10, 10)
    X = center(X)
    nn = build_nn()
    theta = initialize(nn)
