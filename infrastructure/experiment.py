#!/usr/bin/python
# coding=utf-8
"""
The amazing Experiment class i dreamt up recently.
It should be a kind of ML-Experiment-build-system-checkpointer-...
TODO:
 R create Factory for experiments and make constructor simple
 - write report that is readable by humans and this package
 T Test auto-caching
 ! automatic repetition of a stage with mean and var of the result
 - Main should support parameters, loggers, (rnd), many-runs
 - main should also parse command line arguments
 - find out if current file is git-controlled and if it is checked in, warn otherwise
 - write commit hash to report
 ! automatize rerunning an experiment by checking out the appropriate version and feed the parameters
 ? gather versions of dependencies
 V figure out how to incorporate plots
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
from infrastructure.function_helpers import assert_no_duplicate_args, assert_no_unexpected_kwargs, assert_no_missing_args, apply_options

__all__ = ['Experiment']

RANDOM_SEED_RANGE = 0, 1000000

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
        # internal state
        self.execution_time = None

    def add_random_arg_to(self, arguments):
        if 'rnd' in self.signature['args']  and 'rnd' not in arguments:
            arguments['rnd'] = self.random

    def add_logger_arg_to(self, arguments):
        if 'logger' in self.signature['args']:
            arguments['logger'] = self.logger

    def __call__(self, *args, **kwargs):
        # Modify Arguments
        assert_no_unexpected_kwargs(self.signature, kwargs)
        assert_no_duplicate_args(self.signature, args, kwargs)
        arguments = apply_options(self.signature, args, kwargs, self.options)
        self.add_random_arg_to(arguments)
        key = (self.source, dict(arguments)) # use arguments without logger as cache-key
        self.add_logger_arg_to(arguments)
        assert_no_missing_args(self.signature, arguments)
        # Check for cached version
        try:
            result = self.cache[key]
            self.logger.info("Retrieved '%s' from cache. Skipping Execution"%self.__name__)
        except KeyError:
        #### Run the function ####
            start_time = time.time()
            result = self.function(**arguments)
            self.execution_time = time.time() - start_time
            self.logger.info("Completed Stage '%s' in %2.2f sec"%(self.__name__, self.execution_time))
        ##########################
            if self.execution_time > self.caching_threshold:
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