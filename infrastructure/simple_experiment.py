#!/usr/bin/python
# coding=utf-8
"""
# This is a demo experiment to explore the workflow.
# I'll (mis)use this docstring to give a description and also
# contain the configuration file.

seed = 1234567890
n = 4
m = 4
bias = 1

"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import time
from experiment import Experiment, ShelveCache


ex = Experiment( __doc__, cache=ShelveCache("experiment.shelve"))

@ex.stage
def create_data(n, m, bias, logger):
    logger.info("Creating {}x{} array with bias {}.".format(n,m, bias))
    A = np.arange(n*m).reshape(n,m) + bias
    return A

@ex.stage
def square(A, logger):
    logger.info("Squaring {}x{} array.".format(A.shape[0], A.shape[1]))
    return A * A

@ex.main
def main():
    A = create_data()
    B = square(A)

    time.sleep(.1)
    print(B)

