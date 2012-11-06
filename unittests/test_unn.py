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
from scipy.optimize import check_grad, approx_fprime
import unittest

from UNN.connections import LinearCombination
from UNN.error_functions import sum_of_squares_error


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[-1, 1, 0, 1]]).reshape(-1)
        self.X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]])
        self.X_nb = self.X[:,:-1] # no bias included
        self.T = np.array([[1, 0, 2, 1, 1]]).T

    def assert_backprop_correct(self, connection, theta, X, T, epsilon=1e-7):
        Y = connection.forward_pass(theta, X)
        out_error = Y - T
        in_error, grad = connection.backprop(theta, X, Y, out_error)

        func_theta = lambda th : sum_of_squares_error(connection.forward_pass(th, X), T)
        func_x = lambda x, t : sum_of_squares_error(connection.forward_pass(theta, x), t)

        grad_approx = approx_fprime(theta, func_theta, epsilon)
        assert_allclose(grad, grad_approx, atol=1e-5)

        for x, t, e_in in zip(X, T, in_error) :
            in_error_approx = approx_fprime(x, func_x, epsilon, t)
            assert_allclose(e_in, in_error_approx, atol=1e-5)


    def test_LinearCombination_dimensions(self):
        lc = LinearCombination(5, 7)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 7)
        self.assertEqual(lc.get_param_dim(), 5*7)

    def test_LinearCombination_forward_pass_single_samples(self):
        lc = LinearCombination(4, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            self.assertEqual(lc.forward_pass(self.theta, x), t)

    def test_LinearCombination_forward_pass_multi_sample(self):
        lc = LinearCombination(4, 1)
        assert_allclose(lc.forward_pass(self.theta, self.X), self.T)

    def test_LinearCombination_backprop_multisample_zero_is_zero(self):
        lc = LinearCombination(4, 1)
        in_error, grad = lc.backprop(self.theta, self.X, self.T, np.zeros_like(self.T))
        assert_allclose(in_error, np.zeros_like(self.X))
        assert_allclose(grad, np.zeros_like(self.theta))

    def test_LinearCombination_backprop_multisample(self):
        lc = LinearCombination(4, 1)
        self.assert_backprop_correct(lc, self.theta, self.X, np.ones_like(self.T))


if __name__ == '__main__':
    unittest.main()







