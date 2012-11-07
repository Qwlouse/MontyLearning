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

    def forward_pass(self, theta, X_list):
        assert len(theta) == self.get_param_dim()
        in_dim = 0
        in_len = X_list[0].shape[0]
        for x in X_list:
            assert x.shape[0] == in_len
            in_dim += x.shape[1]
        assert in_dim == self.input_dim
        Y = self._forward_pass(theta, X_list)
        assert Y.shape[0] == X_list[0].shape[0]
        assert Y.shape[1] == self.output_dim
        return Y

    def _forward_pass(self, theta, X_list):
        raise NotImplementedError()

    def backprop(self, theta, X, Y, out_error):
        assert len(theta) == self.get_param_dim()
        assert X.shape[1] == self.input_dim
        assert Y.shape[1] == self.output_dim
        assert out_error.shape[1] == self.output_dim
        in_error, grad = self._backprop(theta, X, Y, out_error)
        assert in_error.shape == X.shape
        assert grad.shape == theta.shape
        return in_error, grad

    def _backprop(self, theta, X, Y, out_error):
        raise NotImplementedError()


class AdditiveConnection(Connection):

    def get_param_dim(self):
        return 0

    def _forward_pass(self, theta, X_list):
        i = 0
        Y = np.zeros((X_list[0].shape[0], self.output_dim))
        for x in X_list:
            x_dim = x.shape[1]
            s = slice(i, i+x_dim)
            i += x_dim
            Y += self._split_forward_pass(theta, x, s)
        return Y

    def _split_forward_pass(self, theta, x, part):
        return x



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

    def _split_forward_pass(self, theta, x, part):
        w = self.unpackTheta(theta)[part, :]
        return x.dot(w)

    def _backprop(self, theta, X, Y, out_error):
        W = self.unpackTheta(theta)
        X = np.atleast_2d(X)
        grad = X.T.dot(out_error).flatten()
        in_error = out_error.dot(W.T)
        return in_error, grad


class Sigmoid(Connection):
    def __init__(self, input_dim, output_dim):
        super(Sigmoid, self).__init__(input_dim, output_dim)
        if input_dim != output_dim:
            raise ValueError("Input and output dimensions must match!")


    def get_param_dim(self):
        return 0

    def _forward_pass(self, _, X_list):
        Y = np.zeros((X_list[0].shape[0], self.output_dim))
        i = 0
        for x in X_list:
            x_dim = x.shape[1]
            Y[:, i:i+x_dim] = 1/(1 + np.exp(-x))
            i += x_dim
        return Y

    def _backprop(self, _, X, Y, out_error):

        return out_error * Y * (1-Y), np.array([])