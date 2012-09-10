#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function
import numpy as np
from datasets import load_xor, load_and, generate_majority_vote, load_iris
from helpers import *

def assert_dataset_wellformed(ds):
    assert hasattr(ds, "data"), "Dataset has to have a 'data' attribute"
    if hasattr(ds, "target"):
        data = np.atleast_2d(ds.data)
        target = np.atleast_1d(ds.data)
        assert_equal(data.shape[0], target.shape[0])

def test_load_xor_wellformed() :
    xor_problem = load_xor()
    assert_dataset_wellformed(xor_problem)

def test_load_and_wellformed() :
    and_problem = load_and()
    assert_dataset_wellformed(and_problem)

def test_generate_majority_vote_wellformed():
    vote_problem = generate_majority_vote()
    assert_dataset_wellformed(vote_problem)

def test_load_iris_wellformed():
    iris = load_iris()
    assert_dataset_wellformed(iris)
