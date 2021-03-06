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

from UNN.connections import LinearCombination, Sigmoid, RectifiedLinear
from UNN.containers import SequentialContainerConnection
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

def assert_backprop_correct(connection, theta, X_list, T, epsilon=1e-7):
    Y = np.zeros(T.shape, dtype=T.dtype)
    connection.forward_pass(theta, X_list, Y)
    out_error = Y - T
    in_error_buffers = connection.create_in_error_buffers_like(X_list)
    connection.backprop(theta, X_list, Y, out_error, in_error_buffers)
    grad = connection.create_grad_buf()
    connection.calculate_gradient(theta, grad, X_list, Y, in_error_buffers, out_error)
    out_buf = connection.create_out_buf_like(X_list)
    def func_theta(th):
        connection.forward_pass(th, X_list, out_buf)
        return sum_of_squares_error(out_buf, T)

    def func_x(x):
        connection.forward_pass(theta, [x], out_buf)
        return sum_of_squares_error(out_buf, T)

    grad_approx = approx_fprime(theta, func_theta, epsilon)
    assert_allclose(grad, grad_approx, atol=1e-5)

    stacked_X = np.hstack(tuple(X_list))
    in_error_approx = approx_fprime(stacked_X, func_x, epsilon)
    stacked_in_error = np.hstack(tuple(in_error_buffers))
    assert_allclose(stacked_in_error, in_error_approx, atol=1e-5)



class LinearCombinationTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[-1, 1, 0, 1]], dtype=np.float64).reshape(-1)
        self.X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]], dtype=np.float64)
        self.T = np.array([[1, 0, 2, 1, 1]], dtype=np.float64).T

    def test_dimensions(self):
        lc = LinearCombination(5, 7)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 7)
        self.assertEqual(lc.get_param_dim(), 5*7)

    def test_forward_pass_single_samples(self):
        lc = LinearCombination(4, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            out_buf = np.zeros_like(t)
            lc.forward_pass(self.theta, [x], out_buf)
            self.assertEqual(out_buf, t)

    def test_forward_pass_multi_sample(self):
        lc = LinearCombination(4, 1)
        out_buf = np.zeros(self.T.shape, dtype=self.T.dtype)
        lc.forward_pass(self.theta, [self.X], out_buf)
        assert_allclose(out_buf, self.T)

    def test_backprop_multisample_zero_is_zero(self):
        lc = LinearCombination(4, 1)
        in_error_buffers = [np.zeros_like(self.X)]
        lc.backprop(self.theta, [self.X], self.T, np.zeros_like(self.T), in_error_buffers)
        grad = np.zeros_like(self.theta)
        lc.calculate_gradient(self.theta, grad, [self.X], self.T, in_error_buffers, np.zeros_like(self.T))
        assert_allclose(in_error_buffers[0], np.zeros_like(self.X))
        assert_allclose(grad, np.zeros_like(self.theta))

    def test_backprop_multisample(self):
        lc = LinearCombination(4, 1)
        assert_backprop_correct(lc, self.theta, [self.X], np.ones_like(self.T))


class SigmoidTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[4., 3.5, 3., 2.5, 2., 1.5, 1., 0.5, 0., -0.5, -1.,
                            -1.5, -2., -2.5, -3., -3.5, -4.]]).T
        self.T = np.array([[0.98201379,  0.97068777,  0.95257413,  0.92414182,
                            0.88079708,  0.81757448,  0.73105858,  0.62245933,
                            0.5       ,  0.37754067,  0.26894142,  0.18242552,
                            0.11920292,  0.07585818,  0.04742587,  0.02931223,
                            0.01798621]]).T

    def test_dimensions(self):
        lc = Sigmoid(5, 5)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 5)
        self.assertEqual(lc.get_param_dim(), 0)

    def test_forward_pass_single_samples(self):
        lc = Sigmoid(1, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            out_buf = np.zeros_like(t)
            lc.forward_pass(np.array([]), [x], out_buf)
            assert_allclose(out_buf, t)

    def test_forward_pass_multi_sample(self):
        lc = Sigmoid(1, 1)
        out_buf = np.zeros(self.T.shape, dtype=self.T.dtype)
        lc.forward_pass(np.array([]), [self.X], out_buf)
        assert_allclose(out_buf, self.T)

    def test_backprop_multisample(self):
        lc = Sigmoid(1, 1)
        assert_backprop_correct(lc, np.array([]), [self.X], np.ones_like(self.T))

class RectifiedLinearTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(-4, 4, 0.3).reshape(-1, 1)
        self.T = np.array([max(0., x) for x in self.X]).reshape(-1, 1)

    def test_dimensions(self):
        lc = RectifiedLinear(5, 5)
        self.assertEqual(lc.input_dim, 5)
        self.assertEqual(lc.output_dim, 5)
        self.assertEqual(lc.get_param_dim(), 0)

    def test_forward_pass_single_samples(self):
        lc = RectifiedLinear(1, 1)
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            out_buf = np.zeros_like(t)
            lc.forward_pass(np.array([]), [x], out_buf)
            assert_allclose(out_buf, t)

    def test_forward_pass_multi_sample(self):
        lc = RectifiedLinear(1, 1)
        out_buf = np.zeros(self.T.shape, dtype=self.T.dtype)
        lc.forward_pass(np.array([]), [self.X], out_buf)
        assert_allclose(out_buf, self.T)

    def test_backprop_multisample(self):
        lc = RectifiedLinear(1, 1)
        assert_backprop_correct(lc, np.array([]), [self.X], np.ones_like(self.T))


class SequentialContainerConnectionTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[-1, 1, 0, 1]], dtype=np.float64).reshape(-1)
        self.X = np.array([[0, 0, 0, 1], [1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1],[1, 1, 0, 1]], dtype=np.float64)
        self.T = np.array([[0.73105858], [ 0.5],[0.88079708],[0.73105858],[0.73105858]])

    def test_dimensions(self):
        lc = LinearCombination(5, 7)
        sig = Sigmoid(7, 7)
        scc = SequentialContainerConnection(5, 7, [lc, sig])
        self.assertEqual(scc.input_dim, 5)
        self.assertEqual(scc.output_dim, 7)
        self.assertEqual(scc.get_param_dim(), 5*7)

    def test_forward_pass_single_samples(self):
        lc = LinearCombination(4, 1)
        sig = Sigmoid(1, 1)
        scc = SequentialContainerConnection(4, 1, [lc, sig])
        for x, t in zip(self.X, self.T):
            t = np.atleast_2d(t)
            x = np.atleast_2d(x)
            out_buf = np.zeros_like(t)
            scc.forward_pass(self.theta, [x], out_buf)
            assert_allclose(out_buf, t)

    def test_forward_pass_multi_sample(self):
        lc = LinearCombination(4, 1)
        sig = Sigmoid(1, 1)
        scc = SequentialContainerConnection(4, 1, [lc, sig])
        out_buf = np.zeros(self.T.shape, dtype=self.T.dtype)
        scc.forward_pass(self.theta, [self.X], out_buf)
        assert_allclose(out_buf, self.T)

    def test_backprop_multisample_zero_is_zero(self):
        lc = LinearCombination(4, 1)
        sig = Sigmoid(1, 1)
        scc = SequentialContainerConnection(4, 1, [lc, sig])
        in_error_buffers = [np.zeros_like(self.X)]
        scc.backprop(self.theta, [self.X], self.T, np.zeros_like(self.T), in_error_buffers)
        grad = np.zeros_like(self.theta)
        scc.calculate_gradient(self.theta, grad, [self.X], self.T, in_error_buffers, np.zeros_like(self.T))
        assert_allclose(in_error_buffers[0], np.zeros_like(self.X))
        assert_allclose(grad, np.zeros_like(self.theta))

    def test_backprop_multisample(self):
        lc = LinearCombination(4, 1)
        sig = Sigmoid(1, 1)
        scc = SequentialContainerConnection(4, 1, [lc, sig])
        assert_backprop_correct(scc, self.theta, [self.X], np.ones_like(self.T))


if __name__ == '__main__':
    unittest.main()







