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
        """
        Return the number of parameters of this connection.
        """
        raise NotImplementedError()

    def forward_pass(self, theta, X_list, out_buf):
        """
        Do one forward pass of this connection and write the result to out_buf.

        :param theta: 1d array of parameter values to be used
        :param X_list: list of 2d arrays to be used as input. Each entry in the
                       list should be a N x m array were N is the number of
                       samples and m < input_dim is a subset of the input dims.
        :param out_buf: 2d array with size N x output_dim to which the results
                        get written.
        """
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

    def backprop(self, theta, X_list, Y, out_error, in_error_buffers):
        assert theta.shape == (self.get_param_dim(),)
        in_dim = 0
        in_len = X_list[0].shape[0]
        for x in X_list:
            assert x.shape[0] == in_len
            in_dim += x.shape[1]
        assert Y.shape == (in_len, self.output_dim)
        assert out_error.shape == (in_len, self.output_dim)
        assert len(in_error_buffers) == len(X_list)
        for x, e in zip(X_list, in_error_buffers):
            assert e.shape == x.shape
        self._backprop(theta, X_list, Y, out_error, in_error_buffers)

    def _backprop(self, theta, X_list, Y, out_error, in_error_buffers):
        raise NotImplementedError()

    def calculate_gradient(self, theta, grad_buf, X_list, Y, in_error_list, out_error):
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
        assert grad_buf.shape == theta.shape
        self._calculate_gradient(theta, grad_buf, X_list, Y, in_error_list, out_error)


    def _calculate_gradient(self, theta, grad_buf, X_list, Y, in_error_list, out_error):
        pass


class AdditiveConnection(Connection):
    def get_param_dim(self):
        return 0

    def _forward_pass(self, theta, X_list, out_buf):
        i = 0
        o = np.zeros_like(out_buf)
        out_buf[:] = 0
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            self._split_forward_pass(theta, x, o, s)
            out_buf += o

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = x

    def _backprop(self, theta, X_list, Y, out_error, in_error_buffers):
        i = 0
        for x, in_error_buf in zip(X_list, in_error_buffers):
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            self._split_backprop(theta, x, Y, out_error, in_error_buf, s)

    def _split_backprop(self, theta, x, Y, out_error, in_error_buf, part):
        in_error_buf[:] = out_error



class ConcatenatingConnection(Connection):
    def get_param_dim(self):
        return 0

    def _forward_pass(self, theta, X_list, out_buf):
        i = 0
        out_buf[:] = 0
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            self._split_forward_pass(theta, x,out_buf[:, s], s)

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = x

    def _backprop(self, theta, X_list, Y, out_error, in_error_buffers):
        i = 0
        for x, in_error_buf in zip(X_list, in_error_buffers):
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            e = out_error[:, s]
            y = Y[:, s]
            self._split_backprop(theta, x, y, e, in_error_buf, s)


    def _split_backprop(self, theta, x, y, out_e, in_error_buf, part):
        in_error_buf[:] = out_e



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

    def _split_backprop(self, theta, X, Y, out_error, in_error_buf, part):
        W = self.unpackTheta(theta)
        np.dot(out_error, W.T, out=in_error_buf)

    def _calculate_gradient(self, theta, grad_buf, X_list, Y, in_error_list, out_error):
        i = 0
        for x in X_list:
            g = x.T.dot(out_error).flatten()
            grad_buf[i:i+len(g)] += g
            i += len(g)


class Sigmoid(ConcatenatingConnection):
    def __init__(self, input_dim, output_dim):
        super(Sigmoid, self).__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError("Input and output dimensions must match!")

    def _split_forward_pass(self, theta, x, out_buf, part):
        out_buf[:] = 1/(1 + np.exp(-x))

    def _split_backprop(self, theta, x, y, out_e, in_error_buf, part):
        in_error_buf[:] = out_e * y * (1-y)


class RectifiedLinear(ConcatenatingConnection):
    def __init__(self, input_dim, output_dim):
        super(RectifiedLinear, self).__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError("Input and output dimensions must match!")

    def _split_forward_pass(self, theta, x, out_buf, part):
        np.maximum(x, 0, out=out_buf)

    def _split_backprop(self, theta, x, y, out_e, in_error_buf, part):
        in_error_buf[:] = out_e.copy()
        in_error_buf[y == 0] = 0