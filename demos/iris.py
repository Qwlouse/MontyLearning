#!/usr/bin/python
# coding=utf-8
"""
# config for iris demo
seed = 812590
hidden_size = 5
learning_rate = 0.1

"""
from __future__ import division, print_function, unicode_literals
from infrastructure.caches import ShelveCache
from infrastructure.experiment import createExperiment
from datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from neural_nets.connections import FullConnectionWithBias, SigmoidLayer
from neural_nets.fann import FANN
import matplotlib.pyplot as plt

cache = ShelveCache("iris.shelve")
ex = createExperiment("Iris", config_string=__doc__, cache=cache)

@ex.stage
def binarize_labels(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y), lb

@ex.stage
def create_neural_network(in_size, hidden_size, out_size, rnd, logger):
    logger.info("Creating a NN with {} inputs, {} hidden units, and {} output units.".format(in_size, hidden_size, out_size))
    c0 = FullConnectionWithBias(in_size, hidden_size)
    s0 = SigmoidLayer(hidden_size)
    c1 = FullConnectionWithBias(hidden_size, out_size)
    s1 = SigmoidLayer(out_size)
    nn = FANN([c0, s0, c1, s1])
    theta = rnd.randn(nn.get_param_dim())
    return nn, theta

@ex.stage
def epoch_gradient_descent(fann, theta, X, T, learning_rate, logger):
    grad = fann.calculate_gradient(theta, X, T)
    theta_new = theta - grad * learning_rate
    err = fann.calculate_error(theta_new, X, T)
    logger.info("Error: {}".format(err))
    return err, theta_new

@ex.stage
def many_epochs_decrease_lr(fann, theta, X, T, learning_rate):
    errors = []
    err = 1e100
    lr = learning_rate
    for i in range(1, 4000):
        err_new, theta_new = epoch_gradient_descent(fann, theta, X, T, learning_rate=lr)
        if err_new < err :
            theta = theta_new
            errors.append(err_new)
        else :
            print("---")
            lr /= 2
        err = err_new
    return errors


@ex.main
def main():
    iris = load_iris()
    T, lb = binarize_labels(iris.target)
    nn, theta = create_neural_network(in_size=iris.data.shape[1], out_size=T.shape[1])
    errors = many_epochs_decrease_lr(nn, theta, iris.data, T)

    fig, ax = plt.subplots()
    ax.plot(errors)
    plt.show()



