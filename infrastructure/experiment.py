#!/usr/bin/python
# coding=utf-8
"""
The amazing Experiment class i dreamt up recently.
It should be a kind of ML-Experiment-build-system-checkpointer-...
ROADMAP:
 ### Report
 - write report that is readable by humans and this package
 - it should include the results and all the options as well as the code version
 - make such reports repeatable
 ? maybe have a database to store important facts about experiments,
   so you could easily query what you tried and what resulted

 ### Caching
 - make caching key independent of comments and docstring of the stage
 T Test auto-caching

 ### configuration
 - add logging to configuration
 - add caching to configuration
 V have a kind of config-file-hierarchy so i could define some basic settings
   like paths, logging, caching, ... for my project and experiments only need
   to overwrite some options
 ? maybe even provide means to include other config files?

 ### Stage Repetition
 V automatic repetition of a stage with mean and var of the result
 V make stages easily repeatable with different options
 V make option-sweeps easy
    # maybe like this:
    # with ex.optionset("color") as o:
    #     o.just_call_stage() # and get all the options from the color section
    # for o in ex.optionsets(["color", "gray", "label"]):
    #     o.just_call_stage()
    #
    # We could even implement optionsweep like this
    # for o in ex.optionsweep(["gamma", "C", "lambda"], kernel=["linear", "RBF"])
    #     o.call_stage()

 ### Main method
 - Main should support parameters, loggers, (rnd), many-runs
 - main should also parse command line arguments

 ### Version Control integration
 - find out if current file is git-controlled and if it is checked in, warn otherwise
 - write commit hash to report
 ! automatize rerunning an experiment by checking out the appropriate version and feed the parameters
 ? gather versions of dependencies

 ### Display results
 V should be decoupled from console/pc we are running on
 V figure out how to incorporate plots
 V Make very long stages deliver a stream of data to inspect their behaviour live
 ? maybe start a webserver to watch results
 ? maybe incorporate self-updating plots into ipython-notebook

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
from infrastructure.function_helpers import *

__all__ = ['Experiment', 'createExperiment']

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


def createExperiment(name = "Experiment", config_file=None, config_string=None, logger=None, seed=None, cache=None):
    # setup logging
    if logger is None:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info("No Logger configured: Using generic stdout Logger")

    # reading configuration
    options = ConfigObj(unrepr=True)
    if config_file is not None:
        if isinstance(config_file, basestring) :
            logger.info("Loading config file {}".format(config_file))
            options = ConfigObj(config_file, unrepr=True, encoding="UTF-8")
        elif hasattr(config_file, 'read'):
            logger.info("Reading configuration from file.")
            options = ConfigObj(config_file, unrepr=True, encoding="UTF-8")
    elif config_string is not None:
        logger.info("Reading configuration from string.")
        options = ConfigObj(StringIO(str(config_string)), unrepr=True, encoding="UTF8")

    # get seed for random numbers in experiment
    if seed is None:
        if 'seed' in options:
            seed = options['seed']
        else:
            seed = np.random.randint(*RANDOM_SEED_RANGE)
            logger.warn("No Seed given. Using seed={}. Set in config "
                        "file to repeat experiment".format(seed))

    cache = cache or CacheStub()

    return Experiment(name, logger, options, seed, cache)


class Experiment(object):
    def __init__(self, name, logger, options, seed, cache):
        self.name = name
        self.logger = logger
        self.options = options
        self.seed = seed
        self.prng = np.random.RandomState(self.seed)
        self.cache = cache

        # init stages
        self.stages = OrderedDict()

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