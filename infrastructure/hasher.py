#!/usr/bin/python
# coding=utf-8
"""
docstring
"""
from __future__ import division, print_function, unicode_literals
import pickle

def sshash(obj):
    try:
        return hash(obj)
    except TypeError:
        return hash(pickle.dumps(obj))