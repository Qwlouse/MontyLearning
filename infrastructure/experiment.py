#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from functools import wraps

class Experiment(object):
    def __init__(self):
        self.stages = OrderedDict()
        self.options = {}

    def stage(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            vars = f.func_code.co_varnames
            defaults = f.func_defaults or []
            # construct arguments dict
            arguments = dict()
            # default arguments are weakest
            arguments.update(zip(vars[-len(defaults):], defaults))
            # followed by options
            arguments.update(self.options)
            # passed in keyword arguments
            arguments.update(kwargs)
            # positional arguments are strongest
            arguments.update(zip(vars[:len(args)], args))
            print(arguments)
            return f(**arguments)
        self.stages[f.func_name] = wrapped
        return wrapped
