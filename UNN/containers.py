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
from connections import Connection


class SequentialContainerConnection(Connection):
    def __init__(self, input_dim, output_dim, connections=()):
        super(SequentialContainerConnection, self).__init__(input_dim, output_dim)
        self.connections = connections
        current_dim = self.input_dim
        for c in self.connections:
            assert current_dim == c.input_dim
            current_dim = c.output_dim
        self.theta_slices = self.get_theta_slices()

    def get_theta_slices(self):
        slices = []
        theta_offset = 0
        for c in self.connections:
            param_dim = c.get_param_dim()
            slices.append(slice(theta_offset, theta_offset + param_dim))
            theta_offset += param_dim
        return slices


    def get_param_dim(self):
        dim = 0
        for c in self.connections:
            dim += c.get_param_dim()
        return dim

    def _forward_pass(self, theta, X_list, out_buf):
        N = X_list[0].shape[0]
        in_buffers = [X_list] + [[c.create_out_buf(N)] for c in self.connections[:-1]]
        out_buffers = in_buffers[1:] + [[out_buf]]
        for c, in_buf, out, s in zip(self.connections, in_buffers, out_buffers, self.theta_slices):
            c.forward_pass(theta[s], in_buf, out[0])

    def _backprop(self, theta, X_list, Y, out_error, in_error_buffers):
        # ürg! doing everything twice!
        N = X_list[0].shape[0]
        in_buffers = [X_list] + [[c.create_out_buf(N)] for c in self.connections[:-1]]
        out_buffers = in_buffers[1:] + [[Y]]
        for c, in_buf, out, s in zip(self.connections, in_buffers, out_buffers, self.theta_slices):
            c.forward_pass(theta[s], in_buf, out[0])

        in_errors = [in_error_buffers] + [c.create_in_error_buffers_like(x) for c, x in zip(self.connections[1:], in_buffers[1:])]
        out_errors = in_errors[1:] + [[out_error]]
        for c, in_buf, out, in_err, out_err, s in reversed(zip(self.connections, in_buffers, out_buffers, in_errors, out_errors, self.theta_slices)):
            c.backprop(theta[s], in_buf, out[0], out_err[0], in_err)

    def _calculate_gradient(self, theta, grad_buf, X_list, Y, in_error_list, out_error):
        # ürg! doing everything .... thrice!
        N = X_list[0].shape[0]
        in_buffers = [X_list] + [[c.create_out_buf(N)] for c in self.connections[:-1]]
        out_buffers = in_buffers[1:] + [[Y]]
        for c, in_buf, out, s in zip(self.connections, in_buffers, out_buffers, self.theta_slices):
            c.forward_pass(theta[s], in_buf, out[0])

        in_errors = [in_error_list] + [c.create_in_error_buffers_like(x) for c, x in zip(self.connections[1:], in_buffers[1:])]
        out_errors = in_errors[1:] + [[out_error]]
        for c, in_buf, out, in_err, out_err, s in reversed(zip(self.connections, in_buffers, out_buffers, in_errors, out_errors, self.theta_slices)):
            c.backprop(theta[s], in_buf, out[0], out_err[0], in_err)
            c.calculate_gradient(theta[s], grad_buf[s], in_buf, out[0], in_err, out_err[0])
