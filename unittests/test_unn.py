#!/usr/bin/python
# coding=utf-8
# This file is part of the MLizard library published under the GPL3 license.
# Copyright (C) 2012  Klaus Greff
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
docstring
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import unittest

from UNN.connections import LinearCombination, Sigmoid
from UNN.error_functions import sum_of_squares_error

sigmoid_values = [[4.0, 0.9820137900],
                   [3.5, 0.9706877692],
                   [3.0, 0.9525741268],
                   [2.5, 0.9241418200],
                   [2.0, 0.8807970780],
                   [1.5, 0.8175744762],
                   [1.0, 0.7310585786],
                   [0.5, 0.6224593312],
                   [0.0, 0.5],
                  [-0.5, 0.3775406688],
                  [-1.0, 0.2689414213],
                  [-1.5, 0.1824255238],
                  [-2.0, 0.1192029220],
                  [-2.5, 0.0758581800],
                  [-3.0, 0.0474258731],
                  [-3.5, 0.0293122308],
                  [-4.0, 0.0179862100]]

def approx_fprime(xk, f, epsilon, *args):
    f0 = f(xk, *args)
    grad = np.zeros_like(xk, dtype=np.float64).flatten()
    ei = np.zeros_like(xk, dtype=np.float64).flatten()
    for k in range(len(ei)):
        ei[k] = epsilon
        eis = ei.reshape(*xk.shape)
        grad[k] = (f(xk + eis, *args) - f0)/epsilon
        ei[k] = 0.0
    return grad.reshape(*xk.shape)

def assert_backprop_correct(connection, theta, X, T, epsilon=1e-7):
    Y = connection.forward_pass(theta, [X])
    out_error = Y - T
    in_error, grad = connection.backprop(theta, X, Y, out_error)

    func_theta = lambda th : sum_of_squares_error(connection.forward_pass(th, [X]), T)
    func_x = lambda x : sum_of_squares_error(connection.forward_pass(theta, [x]), T)

    grad_approx = approx_fprime(theta, func_theta, epsilon)
    assert_allclose(grad, grad_approx, atol=1e-5)

    in_error_approx = approx_fprime(X, func_x, epsilon)
    assert_allclose(in_error, in_error_approx, atol=1e-5)



class LinearCombinationTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
        self.X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
        self.X_nb = self.X[:,:-1] # no bias included
        self.T = np.array([[1, 0, 2, 1, 1]]).T

    def test_LinearCombination_dimensions(self):
        lc = LinearCombination(5, 7)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 7)
        self.assertEqual(lc.get_param_dim(), 5*7)

    def test_LinearCombination_forward_pass_single_samples(self):
        lc = LinearCombination(4, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            self.assertEqual(lc.forward_pass(self.theta, [x]), t)

    def test_LinearCombination_forward_pass_multi_sample(self):
        lc = LinearCombination(4, 1)
        assert_allclose(lc.forward_pass(self.theta, [self.X]), self.T)

    def test_LinearCombination_backprop_multisample_zero_is_zero(self):
        lc = LinearCombination(4, 1)
        in_error, grad = lc.backprop(self.theta, self.X, self.T, np.zeros_like(self.T))
        assert_allclose(in_error, np.zeros_like(self.X))
        assert_allclose(grad, np.zeros_like(self.theta))

    def test_LinearCombination_backprop_multisample(self):
        lc = LinearCombination(4, 1)
        assert_backprop_correct(lc, self.theta, self.X, np.ones_like(self.T))


class SigmoidTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[4., 3.5, 3., 2.5, 2., 1.5, 1., 0.5, 0., -0.5, -1.,
                            -1.5, -2., -2.5, -3., -3.5, -4.]]).T
        self.T = np.array([[0.98201379,  0.97068777,  0.95257413,  0.92414182,
                            0.88079708,  0.81757448,  0.73105858,  0.62245933,
                            0.5       ,  0.37754067,  0.26894142,  0.18242552,
                            0.11920292,  0.07585818,  0.04742587,  0.02931223,
                            0.01798621]]).T

    def test_Sigmoid_dimensions(self):
        lc = Sigmoid(5, 5)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 5)
        self.assertEqual(lc.get_param_dim(), 0)

    def test_Sigmoid_forward_pass_single_samples(self):
        lc = Sigmoid(1, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            assert_allclose(lc.forward_pass(np.array([]), [x]), t)

    def test_Sigmoid_forward_pass_multi_sample(self):
        lc = Sigmoid(1, 1)
        assert_allclose(lc.forward_pass(np.array([]), [self.X]), self.T)

    def test_Sigmoid_backprop_multisample(self):
        lc = Sigmoid(1, 1)
        assert_backprop_correct(lc, np.array([]), self.X, np.ones_like(self.T))


if __name__ == '__main__':
    unittest.main()






