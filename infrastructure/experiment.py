#!/usr/bin/python
# coding=utf-8
"""
The amazing Experiment class i dreamt up recently.
It should be a kind of ML-Experiment-build-system-checkpointer-...
TODO:
 - write report that is readable by humans and this package
 ! Test auto-caching
 ! automatic repetition of a stage with mean and var of the result
 - Main should support parameters, loggers, (rnd), many-runs
 - main should also parse command line arguments
 - find out if current file is git-controlled and if it is checked in, warn otherwise
 - write commit hash to report
 ! automatize rerunning an experiment by checking out the appropriate version and feed the parameters
 ? gather versions of dependencies
 - figure out how to incorporate plots
"""

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict # TODO: Check if this is necessary
from configobj import ConfigObj
import inspect
import numpy as np
import logging
import time
import os
from StringIO import StringIO
from infrastructure.caches import CacheStub

__all__ = ['Experiment']

RANDOM_SEED_RANGE = 0, 1000000


def get_signature(f):
    args, varargs_name, kw_wildcard_name, defaults = inspect.getargspec(f)
    defaults = defaults or []
    pos_args = args[:len(args)-len(defaults)]
    kwargs = dict(zip(args[-len(defaults):], defaults))
    return {'args' : args,
            'positional' : pos_args,
            'kwargs' : kwargs,
            'varargs_name' : varargs_name,
            'kw_wildcard_name' : kw_wildcard_name}

class StageFunction(object):
    def __init__(self, name, f, cache, options, logger, seed):
        self.function = f
        self.logger = logger
        self.random = np.random.RandomState(seed)
        self.cache = cache
        self.options = options
        # preserve meta_information
        self.__name__ = name
        self.func_name = name
        self.__doc__ = f.__doc__
        # extract extra info
        self.source = str(inspect.getsource(f))
        self.signature = get_signature(f)
        # some configuration
        self.caching_threshold = 2 # seconds

    def apply_options(self, args, kwargs, options):
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
        arguments = dict()
        arguments.update(self.signature['kwargs']) # weakest: default arguments:
        arguments.update((v, options[v]) for v in self.signature['args'] if v in options) # options
        arguments.update(kwargs) # keyword arguments
        positional_arguments = dict(zip(self.signature['positional'], args))
        arguments.update(positional_arguments) # strongest: positional arguments
        return arguments

    def assert_no_missing_args(self, arguments):
        # check if after all some arguments are still missing
        missing_arguments = [v for v in self.signature['args'] if v not in arguments]
        if missing_arguments :
            raise TypeError("{}() is missing value(s) for {}".format(self.__name__, missing_arguments))

    def assert_no_unexpected_kwargs(self, kwargs):
        # check for erroneous kwargs
        wrong_kwargs = [v for v in kwargs if v not in self.signature['args']]
        if wrong_kwargs :
            raise TypeError("{}() got unexpected keyword argument(s): {}".format(self.__name__, wrong_kwargs))

    def assert_no_duplicate_args(self, args, kwargs):
        #check for multiple explicit arguments
        positional_arguments = self.signature['positional'][:len(args)]
        duplicate_arguments = [v for v in positional_arguments if v in kwargs]
        if duplicate_arguments :
            raise TypeError("{}() got multiple values for argument(s) {}".format(self.__name__, duplicate_arguments))

    def add_random_arg_to(self, arguments):
        if 'rnd' in self.signature['args']  and 'rnd' not in arguments:
            arguments['rnd'] = self.random

    def add_logger_arg_to(self, arguments):
        if 'logger' in self.signature['args']:
            arguments['logger'] = self.logger

    def __call__(self, *args, **kwargs):
        # Modify Arguments
        self.assert_no_unexpected_kwargs(kwargs)
        self.assert_no_duplicate_args(args, kwargs)
        arguments = self.apply_options(args, kwargs, self.options)
        self.add_random_arg_to(arguments)
        key = (self.source, dict(arguments)) # use arguments without logger as cache-key
        self.add_logger_arg_to(arguments)
        self.assert_no_missing_args(arguments)
        # Check for cached version
        try:
            result = self.cache[key]
            self.logger.info("Retrieved '%s' from cache. Skipping Execution"%self.__name__)
        except KeyError:
        #### Run the function ####
            start_time = time.time()
            result = self.function(**arguments)
            duration = time.time() - start_time
            self.logger.info("Completed Stage '%s' in %2.2f sec"%(self.__name__, duration))
        ##########################
            if duration > self.caching_threshold:
                self.logger.info("Execution took more than %2.2f sec so we cache the result."%self.caching_threshold)
                self.cache[key] = result
        return result

    def __hash__(self):
        return hash(self.source)


class Experiment(object):
    def __init__(self, config=None, cache=None, seed=None, logger=None):
        self.setup_logging(logger)

        # load options from config
        if isinstance(config, basestring) :
            if os.path.exists(config):
                self.logger.info("Loading config file {}".format(config))
                self.options = ConfigObj(config, unrepr=True, encoding="UTF-8")
            else :
                self.logger.info("Reading configuration from string.")
                self.options = ConfigObj(StringIO(str(config)), unrepr=True, encoding="UTF8")
        elif hasattr(config, 'read'):
            self.logger.info("Reading configuration from file.")
            self.options = ConfigObj(config.split('\n'))
        else:
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
        self.cache = cache or CacheStub()

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


    def stage(self, f):
        """
        Decorator, that converts the function into a stage of this experiment.
        The stage times the execution.

        The stage fills in arguments such that:
        - the original explicit call arguments are preserved
        - missing arguments are filled in by name using options (if possible)
        - default arguments are overridden by options
        - a special 'rnd' parameter is provided containing a
        deterministically seeded numpy.random.RandomState
        - a special 'logger' parameter is provided containing a child of
        the experiment logger with the name of the decorated function
        Errors are still thrown if:
        - you pass an unexpected keyword argument
        - you provide multiple values for an argument
        - after all the filling an argument is still missing"""
        if isinstance(f, StageFunction): # do nothing if it is already a stage
            # TODO: do we need to allow beeing stage of multiple experiments?
            return f
        else :
            stage_name = f.func_name
            stage_logger = self.logger.getChild(stage_name)
            stage_seed = self.prng.randint(*RANDOM_SEED_RANGE)
            stage = StageFunction(stage_name, f, self.cache, self.options, stage_logger, stage_seed)
            self.stages[stage_name] = stage
            return stage


    def main(self, f):
        if f.__module__ == "__main__":
            f()
        return f