#!/usr/bin/python
# coding=utf-8
"""
The amazing Experiment class i dreamt up recently.
It should be a kind of ML-Experiment-build-system-checkpointer-...
TODO:
 - provide special logger parameter
 - time stage execution
 - write report
 - save results to disk if stage is costly and load them next time
 - automatic repetition of a stage with mean and var of the result
 - add notion of a run, i.e. executing many stages together
   but tying them together should be highly customizable (ideally just write a method)
   and support many of the stage features too (parameters, loggers, (rnd), many-runs
 - make init take also strings and file-objects for configuration
"""

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict # TODO: Check if this is necessary
from configobj import ConfigObj
from functools import wraps
import numpy as np
import logging

RANDOM_SEED_RANGE = 0, 1000000


class Experiment(object):
    def __init__(self, filename=None, seed=None, logger=None):
        self.setup_logging(logger)


        # load options from config file
        if filename is not None:
            self.logger.info("Loading config file {}".format(filename))
            self.options = ConfigObj(filename, unrepr=True)
        else :
            self.options = ConfigObj(unrepr=True)

        # get seed for random numbers in experiment
        if seed is not None:
            self.seed = seed
        elif 'seed' in self.options:
            self.seed = self.options['seed']
        else:
            self.seed = np.random.randint(*RANDOM_SEED_RANGE)
            self.logger.warn("No Seed given. Using seed={}. Set in config "
                             "file to repeat experiment".format(self.seed))
        self.prng = np.random.RandomState(self.seed)

        # init stages
        self.stages = OrderedDict()

    def setup_logging(self, logger):
        # setup logging
        if logger is not None:
            self.logger = logger
        else :
            self.logger = logging.getLogger("Experiment")
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
            ch.setFormatter(formatter)
            # add ch to logger
            self.logger.addHandler(ch)

            self.logger.info("No Logger configured: Using generic Experiment Logger")


    def construct_arguments(self, f, args, kwargs):
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
        arguments.update((v, self.options[v]) for v in vars if v in self.options)
        arguments.update(kwargs)
        arguments.update(positional_arguments) # strongest

        # special rnd argument if it wasn't passed manually
        if 'rnd' in vars and \
           'rnd' not in positional_arguments and \
           'rnd' not in kwargs :
            rnd = np.random.RandomState(self.prng.randint(*RANDOM_SEED_RANGE))
            arguments['rnd'] = rnd

        # check if after all some arguments are still missing
        missing_arguments = [v for v in vars if v not in arguments]
        if missing_arguments :
            raise TypeError("{}() is missing value(s) for {}".format(f.func_name, missing_arguments))
        return arguments



    def stage(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            arguments = self.construct_arguments(f, args, kwargs)
            return f(**arguments)
        self.stages[f.func_name] = wrapped
        return wrapped
