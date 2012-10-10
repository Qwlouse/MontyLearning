#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
# stolen from :
# http://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python
# https://github.com/sequenceGeek/cgAutoCast/blob/master/cgAutoCast.py
def boolify(s):
    if s == 'True' or s == 'true':
        return True
    if s == 'False' or s == 'false':
        return False
    raise ValueError('Not Boolean Value!')

def noneify(s):
    """ for None type"""
    if s == 'None':
        return None
    raise ValueError('Not None Value!')


def estimateTypedValue(var):
    """guesses the str representation of the variable's type"""
    #dont need to guess type if it is already un-str typed (not coming from CLI)
    if type(var) != type('aString'):
        return var

    #guess string representation, will default to string if others dont pass
    for caster in (boolify, int, float, noneify, str):
        try:
            return caster(var)
        except ValueError:
            pass

def autocast(dFxn):
    def wrapped(*c, **d):
        cp = [estimateTypedValue(x) for x in c]
        dp = dict( (i, estimateTypedValue(j)) for (i,j) in d.items())
        return dFxn(*cp, **dp)
    return wrapped

