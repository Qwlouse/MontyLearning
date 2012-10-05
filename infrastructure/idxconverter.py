#!/usr/bin/python
# coding=utf-8
"""
Convert IDX files to numpy and back.
http://yann.lecun.com/exdb/mnist/  # bottom of the page
"""
from __future__ import division, print_function, unicode_literals
import numpy as np

TYPE_DICT = {
    0x8 : np.uint8,
    0x9 : np.int8,
    0xB : np.int16,
    0xC : np.int32,
    0xD : np.float32,
    0xE : np.float64
}

def open_idx_file(filename):
    with open(filename, 'rb') as f:
        zero = np.fromfile(f, np.int16, 1).byteswap()[0]
        assert zero == 0, "File should start with two zero-bytes but was %s."%hex(zero)
        type_code = np.fromfile(f, np.uint8, 1)[0]
        assert type_code in TYPE_DICT, "Invalid type code %s."%hex(type_code)
        dim_count = np.fromfile(f, np.uint8, 1)[0]
        shape = np.fromfile(f, np.int32, dim_count).byteswap()
        size = reduce(np.multiply, shape)
        return np.fromfile(f, TYPE_DICT[type_code], size).byteswap().reshape(shape)

def write_idx_file(filename):
    pass
