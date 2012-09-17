#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function

def seqEnum(dataset):
    """
    Generator function that takes a sequential dataset as input and returns
    in each iteration one of the sequences as a tuple of (X, T) such that T
    is a prediction of X.
    """
    for s_i in dataset.seqs:
        seq = dataset.data[s_i]
        X = seq[:-1]
        T = seq[1:]
        yield X, T