#!/usr/bin/python
# coding=utf-8
"""
The amazing Experiment class i dreamt up recently.
It should be a kind of ML-Experiment-build-system-checkpointer-...
TODO:
 - load config file to initialize options in experiment
 - provide special rnd parameter with deterministic seeding
 - provide special logger parameter
 - time stage execution
 - write report
 - save results to disk if stage is costly and load them next time
 - automatic repetition of a stage with mean and var of the result
 - add notion of a run, i.e. executing many stages together
   but tying them together should be highly customizable (ideally just write a method)
   and support many of the stage features too (parameters, loggers, (rnd), many-runs
"""

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict # TODO: Check if this is necessary
from configobj import ConfigObj
from functools import wraps

def construct_arguments(f, options, args, kwargs):
    """
    For a given function f and the *args and **kwargs it was called with,
    construct a new dictionary of arguments such that:
      - the original explicit call arguments are preserved
      - missing arguments are filled in by name using options (if possible)
      - default arguments are overridden by options
      - sensible errors are thrown if:
        * you pass an unexpected keyword argument
        * you provide multiple values for an argument
        * after all the filling an argument is still missing
    """
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
    def __init__(self, filename=None):
        if filename is not None:
            self.options = ConfigObj(filename, unrepr=True)
        else :
            self.options = ConfigObj(unrepr=True)
        self.stages = OrderedDict()


    def stage(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            arguments = construct_arguments(f, self.options, args, kwargs)
            return f(**arguments)
        self.stages[f.func_name] = wrapped
        return wrapped
