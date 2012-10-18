#!/usr/bin/python
# -*- coding: utf-8 -*-
import re


def wholeFieldPredicate(field):
    """return the whole field"""
    if field:
        return (field, )
    else:
        return ()


def tokenFieldPredicate(field):
    """returns the tokens"""
    return tuple(field.split())


def commonIntegerPredicate(field):
    """"return any integers"""
    return tuple(re.findall("\d+", field))


def nearIntegersPredicate(field):
    """return any integers N, N+1, and N-1"""
    ints = sorted([int(i) for i in re.findall("\d+", field)])
    return tuple([(i - 1, i, i + 1) for i in ints])


def commonFourGram(field):
    """return 4-grams"""
    return tuple([field[pos:pos + 4] for pos in xrange((len(field) - 4 + 1))])



def commonSixGram(field):
    """"return 6-grams"""
    return tuple([field[pos:pos + 6] for pos in xrange((len(field) - 6 + 1))])


def sameThreeCharStartPredicate(field):
    """return first three characters"""
    if len(field) < 3:
        return ()

    return (field[:3], )


def sameFiveCharStartPredicate(field):
    """return first five characters"""
    if len(field) < 5:
        return ()

    return (field[:5], )


def sameSevenCharStartPredicate(field):
    """return first seven charactesr"""
    if len(field) < 7:
        return ()

    return (field[:7], )
