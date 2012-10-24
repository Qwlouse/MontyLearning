#!/usr/bin/python
# coding=utf-8
"""
# This is a demo experiment to explore the workflow.
# I'll (mis)use this docstring to give a description and also
# contain the configuration file.

seed = 1234567
hidden_units = 10
iterations = 1000
learning_rate = 0.01

"""
from __future__ import division, print_function, unicode_literals
from mlizard.experiment import createExperiment
from neural_nets.connections import FullConnectionWithBias, SigmoidLayer
from neural_nets.fann import FANN
from datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

ex = createExperiment("demo", config_string=__doc__)

@ex.stage
def binarize_labels(y):
    lb = LabelBinarizer()
    return lb, lb.fit_transform(y)

@ex.stage
def build_nn(hidden_units):
    l0 = FullConnectionWithBias(4, hidden_units)
    s0 = SigmoidLayer(hidden_units)
    l1 = FullConnectionWithBias(hidden_units, 3)
    s1 = SigmoidLayer(3)
    return FANN([l0, s0, l1, s1])

@ex.stage
def initialize(nn, rnd):
    return rnd.randn(nn.get_param_dim())

@ex.stage
def gradient_descent(nn, theta_start, X, T, iterations, learning_rate, logger):
    theta = theta_start.copy()
    for i in range(iterations):
        grad = nn.calculate_gradient(theta, X, T)
        theta -= grad * learning_rate
        if i % 100 == 0 :
            error = nn.calculate_error(theta, X, T)
            logger.debug("iteration {} error: {}".format(i, error))
    return theta

@ex.main
def main():
    # load data
    iris = load_iris()
    lb, y = binarize_labels(iris.target)    # stage 1
    nn = build_nn()                         # stage 2
    theta_start = initialize(nn)                  # stage 3
    theta = gradient_descent(nn, theta_start, iris.data, y)


