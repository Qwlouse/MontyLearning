#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function

from numpy.testing import assert_almost_equal, assert_equal
from numpy.testing import assert_array_less as assert_less
from nose.tools import istest, nottest, with_setup, assert_true

# Use the same flag as unittest itself to prevent descent into these functions:
__unittest = 1

__all__ = ["assert_almost_equal", "assert_equal", "istest", "nottest", "with_setup", "assert_true"]