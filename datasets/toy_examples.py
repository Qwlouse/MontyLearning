#!/usr/bin/python
# coding: utf-8
from __future__ import division, unicode_literals, print_function

from bunch import Bunch
import numpy as np


def load_xor():
    xor = Bunch()
    xor.DESCR = "The XOR function from logic. A Toy Example for Neural Networks. "\
                "Needs at least two hidden units."
    xor.data = np.array([[0,0],[0,1],[1,0],[1,1]])
    xor.target = np.array([[0, 1, 1, 0]]).T
    return xor

def load_and():
    and_ = Bunch()
    and_.DESCR = "The AND function from logic. A Toy Example for Neural Networks. "
    and_.data = np.array([[0,0],[0,1],[1,0],[1,1]])
    and_.target = np.array([[0, 1, 1, 1]]).T
    return and_

def generate_majority_vote(n = 200, m = 9):
    vote = Bunch()
    vote.DESCR = "Toy example to train a network how to do a majority vote."\
                 "Target is 1 iff there are more 1s than 0s in the input."
    vote.data = np.random.randint(0, 2, (n, m))
    vote.target = (np.sum(vote.data, 1) > m // 2).reshape(-1,1)
    return vote