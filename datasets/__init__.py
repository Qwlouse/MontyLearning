#!/usr/bin/python
# coding: utf-8
"""
Datasets are presented in the scikits.learn format. That means a Bunch containing:
  * data   : ndarray with one sample per line and one feature per column
  * target : ndarray with one output sample per line (for supervised tasks)
  * DESCR  : description of the dataset (optional)
"""

from __future__ import division, unicode_literals, print_function

# Standard Datasets from sklearn
from scikits.learn.datasets import load_iris

from toy_examples import *