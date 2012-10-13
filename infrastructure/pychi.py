#!/usr/bin/python
# coding=utf-8
"""
Inspiriert von Perl CHI:
http://search.cpan.org/~jswartz/CHI-0.55/lib/CHI.pm
http://www.youtube.com/watch?v=9fb3kpzOWxI
"""
from __future__ import division, print_function, unicode_literals
from infrastructure.hasher import sshash
from cPickle import dumps, loads

class Cache(object):
    def __init__(self, driver, hash = sshash, serialize=dumps, deserialize=loads):
        """
        driver: dict-like object
                (i.e. supports __getitem__, __setitem__, __delitem__, clear)
        hash: function that can hash arbitrary(!) objects. defaults to sshash
        serialize/deserialize : functions that are used to serialize/
                                deserialize the values defaults to
                                cPickle.dumps and cPickle.loads
        """
        self.driver = driver
        self.hash = hash
        self.serialize = serialize
        self.deserialize = deserialize


    def __getitem__(self, key):
        hash = self.hash(key)
        return self.deserialize(self.driver[hash])

    def __setitem__(self, key, value):
        hash = self.hash(key)
        self.driver[hash] = self.serialize(value)

    def __delitem__(self, key):
        hash = self.hash(key)
        del self.driver[hash]

    def clear(self):
        self.driver.clear()

    def compute(self, key, function):
        try:
            return self[key]
        except KeyError:
            result = function()
            self[key] = result
            return result
