#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from functools import wraps

def construct_arguments(f, options, args, kwargs):
    vars = f.func_code.co_varnames
    # check for erroneous kwargs
    wrong_kwargs = [v for v in kwargs if v not in vars]
    if wrong_kwargs :
        raise TypeError("{}() got unexpected keyword argument(s): {}".format(f.func_name, wrong_kwargs))

    defaults = f.func_defaults or []
    default_arguments = dict(zip(vars[-len(defaults):], defaults))
    positional_arguments = dict(zip(vars[:len(args)], args))

    #check for multiple explicit arguments
    duplicate_arguments = [v for v in vars if v in positional_arguments and v in kwargs]
    if duplicate_arguments :
            raise TypeError("{}() got multiple values for argument(s) {}".format(f.func_name, duplicate_arguments))

    arguments = dict()
    arguments.update(default_arguments) # weakest
    arguments.update((v, options[v]) for v in vars if v in options)
    arguments.update(kwargs)
    arguments.update(positional_arguments) # strongest

    # check if after all some arguments are still missing
    missing_arguments = [v for v in vars if v not in arguments]
    if missing_arguments :
            raise TypeError("{}() is missing value(s) for {}".format(f.func_name, missing_arguments))
    return arguments


class Experiment(object):
    def __init__(self):
        self.stages = OrderedDict()
        self.options = {}

    def stage(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            arguments = construct_arguments(f, self.options, args, kwargs)
            return f(**arguments)
        self.stages[f.func_name] = wrapped
        return wrapped
