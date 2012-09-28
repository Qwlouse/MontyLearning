#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from helpers import *
from infrastructure.experiment import Experiment


def test_Experiment_constructor_works():
    ex1 = Experiment()

def test_Experiment_provides_stage_decorator():
    ex1 = Experiment()

    @ex1.stage
    def foo(): pass

    assert_true(hasattr(foo, '__call__'))

def test_stage_decorator_retains_docstring():
    ex1 = Experiment()

    @ex1.stage
    def foo():
        """
        Test-Docstring
        """
        pass

    assert_equal(foo.__doc__.strip(), "Test-Docstring")


def test_stage_decorator_retains_function_name():
    ex1 = Experiment()

    @ex1.stage
    def foo(): pass

    assert_equal(foo.func_name, "foo")

def test_Experiment_keeps_track_of_stages():
    ex1 = Experiment()

    @ex1.stage
    def foo(): pass

    @ex1.stage
    def bar(): pass

    @ex1.stage
    def baz(): pass

    assert_equal(ex1.stages["foo"], foo)
    assert_equal(ex1.stages["bar"], bar)
    assert_equal(ex1.stages["baz"], baz)


def test_Experiment_preserves_order_of_stages():
    ex1 = Experiment()

    @ex1.stage
    def foo(): pass

    @ex1.stage
    def bar(): pass

    @ex1.stage
    def baz(): pass

    assert_equal(ex1.stages.keys(), ["foo", "bar", "baz"])

def test_stage_executes_function():
    ex1 = Experiment()
    a = []

    @ex1.stage
    def foo():
        a.append("executed")

    foo()
    assert_equal(a, ["executed"])

def test_Experiment_stores_options():
    ex1 = Experiment()
    ex1.options["alpha"] = 0.7
    assert_equal(ex1.options["alpha"], 0.7)

def test_stage_applies_options():
    ex1 = Experiment()
    ex1.options["alpha"] = 0.7
    ex1.options["beta"] = 1.2

    @ex1.stage
    def foo(alpha, beta):
        return alpha, beta

    #noinspection PyArgumentList
    assert_equal(foo(), (0.7, 1.2))

def test_stage_overrides_default_with_options():
    ex1 = Experiment()
    ex1.options["alpha"] = 0.7
    ex1.options["beta"] = 1.2

    @ex1.stage
    def foo(alpha=0, beta=0):
        return alpha, beta

    assert_equal(foo(), (0.7, 1.2))

def test_stage_keeps_explicit_arguments():
    ex1 = Experiment()
    ex1.options["alpha"] = 0.7
    ex1.options["beta"] = 1.2

    @ex1.stage
    def foo(alpha, beta):
        return alpha, beta

    assert_equal(foo(0, beta=0), (0, 0))

@raises(TypeError)
def test_stage_with_unexpected_kwarg_raises_TypeError():
    ex1 = Experiment()

    @ex1.stage
    def foo(): pass

    #noinspection PyArgumentList
    foo(unexpected=1)

@raises(TypeError)
def test_stage_with_duplicate_arguments_raises_TypeError():
    ex1 = Experiment()

    @ex1.stage
    def foo(a): pass

    #noinspection PyArgumentList
    foo(2, a=1)

@raises(TypeError)
def test_stage_with_missing_arguments_raises_TypeError():
    ex1 = Experiment()
    ex1.options["b"]=1
    @ex1.stage
    def foo(a, b, c, d=5): pass

    #noinspection PyArgumentList
    foo(1)