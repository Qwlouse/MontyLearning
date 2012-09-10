#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function

import numpy as np
from neural_nets.functions import sigmoid, sigmoid_dx
from numpy.testing import assert_almost_equal
################## Sigmoid ###########################

known_values = { 4.0: 0.9820137900,
                 3.5: 0.9706877692,
                 3.0: 0.9525741268,
                 2.5: 0.9241418200,
                 2.0: 0.8807970780,
                 1.5: 0.8175744762,
                 1.0: 0.7310585786,
                 0.5: 0.6224593312,
                 0.0: 0.5,
                 -0.5: 0.3775406688,
                 -1.0: 0.2689414213,
                 -1.5: 0.1824255238,
                 -2.0: 0.1192029220,
                 -2.5: 0.0758581800,
                 -3.0: 0.0474258731,
                 -3.5: 0.0293122308,
                 -4.0: 0.0179862100}

def test_sigmoid_some_numbers():
    for x, y in known_values.items():
        assert_almost_equal(sigmoid(x), y)

def test_sigmoid_np_arrays():
    X = np.array(known_values.keys())
    Y = np.array(known_values.values())
    assert_almost_equal(sigmoid(X), Y)

def test_sigmoid_dx_gradient():
    epsilon = 1e-4
    for x in np.arange(-4, 4.5, 0.5):
        sl = sigmoid(x-epsilon)
        sr = sigmoid(x+epsilon)
        s_dx = (sr - sl) / (2 * epsilon)
        assert_almost_equal(sigmoid_dx(x), s_dx)