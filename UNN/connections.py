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


class Connection(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def get_param_dim(self):
        raise NotImplementedError()

    def forward_pass(self, theta, X_list, out_buf):
        assert theta.shape == (self.get_param_dim(),)
        in_dim = 0
        in_len = X_list[0].shape[0]
        for x in X_list:
            assert x.shape[0] == in_len
            in_dim += x.shape[1]
        assert in_dim == self.input_dim
        assert out_buf.shape == (in_len, self.output_dim)
        self._forward_pass(theta, X_list, out_buf)

    def _forward_pass(self, theta, X_list, out_buf):
        raise NotImplementedError()

    def backprop(self, theta, X_list, Y, out_error):
        assert theta.shape == (self.get_param_dim(),)
        in_dim = 0
        in_len = X_list[0].shape[0]
        for x in X_list:
            assert x.shape[0] == in_len
            in_dim += x.shape[1]
        assert Y.shape == (in_len, self.output_dim)
        assert out_error.shape == (in_len, self.output_dim)
        in_error= self._backprop(theta, X_list, Y, out_error)
        assert len(in_error) == len(X_list)
        for x, e in zip(X_list, in_error):
            assert e.shape == x.shape
        return in_error

    def _backprop(self, theta, X_list, Y, out_error):
        raise NotImplementedError()

    def calculate_gradient(self, theta, X_list, Y, in_error_list, out_error):
        assert theta.shape == (self.get_param_dim(),)
        in_dim = 0
        in_len = X_list[0].shape[0]
        for x in X_list:
            assert x.shape[0] == in_len
            in_dim += x.shape[1]
        assert Y.shape == (in_len, self.output_dim)
        assert out_error.shape == (in_len, self.output_dim)
        assert len(in_error_list) == len(X_list)
        for x, e in zip(X_list, in_error_list):
            assert e.shape == x.shape
        grad = self._calculate_gradient(theta, X_list, Y, in_error_list, out_error)
        assert grad.shape == theta.shape
        return grad


    def _calculate_gradient(self, theta, X_list, Y, in_error_list, out_error):
        raise NotImplementedError()


class AdditiveConnection(Connection):
    def get_param_dim(self):
        return 0

    def _forward_pass(self, theta, X_list, out_buf):
        i = 0
        o = np.zeros_like(out_buf)
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            self._split_forward_pass(theta, x, o, s)
            out_buf += o

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = x

    def _backprop(self, theta, X_list, Y, out_error):
        i = 0
        in_error = []
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            in_error.append(self._split_backprop(theta, x, Y, out_error, s))
        return in_error

    def _split_backprop(self, theta, x, Y, out_error, part):
        return out_error

    def _calculate_gradient(self, theta, X_list, Y, in_error_list, out_error):
        return np.array([])



class ConcatenatingConnection(Connection):
    def get_param_dim(self):
        return 0

    def _forward_pass(self, theta, X_list, out_buf):
        i = 0
        out_buf = 0
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            self._split_forward_pass(theta, x,out_buf[:, s], s)

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = x

    def _backprop(self, theta, X_list, Y, out_error):
        i = 0
        in_error = []
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            e = out_error[:, s]
            y = Y[:, s]
            in_error.append(self._split_backprop(theta, x, y, e, s))
        return in_error

    def _split_backprop(self, theta, x, y, out_e, part):
        return out_e

    def _calculate_gradient(self, theta, X_list, Y, in_error_list, out_error):
        return np.array([])


class LinearCombination(AdditiveConnection):
    """
    Full feed-forward connection without bias.
    """
    def get_param_dim(self):
        """
        Return the dimension of the parameter-space.
        """
        return self.input_dim * self.output_dim

    def unpackTheta(self, theta):
        return theta.reshape(self.input_dim, self.output_dim)

    def _split_forward_pass(self, theta, x, out_buf, part):
        w = self.unpackTheta(theta)[part, :]
        np.dot(x, w, out=out_buf)

    def _split_backprop(self, theta, X, Y, out_error, part):
        W = self.unpackTheta(theta)
        #
        in_error = out_error.dot(W.T)
        return in_error

    def _calculate_gradient(self, theta, X_list, Y, in_error_list, out_error):
        grad = np.zeros_like(theta)
        i = 0
        for x in X_list:
            g = x.T.dot(out_error).flatten()
            grad[i:i+len(g)] = g
            i += len(g)
        return grad


class Sigmoid(ConcatenatingConnection):
    def __init__(self, input_dim, output_dim):
        super(Sigmoid, self).__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError("Input and output dimensions must match!")

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = 1/(1 + np.exp(-x))

    def _split_backprop(self, theta, x, y, out_e, part):
        return out_e * y * (1-y)


class RectifiedLinear(ConcatenatingConnection):
    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError("Input and output dimensions must match!")

    def _split_forward_pass(self, theta, x, out_buf, part):
        np.maximum(x, 0, out=out_buf)

    def _split_backprop(self, theta, x, y, out_e, part):
        in_e = out_e.copy()
        in_e[y == 0] = 0
        return in_e