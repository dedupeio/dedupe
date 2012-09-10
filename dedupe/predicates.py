#!/usr/bin/python
# -*- coding: utf-8 -*-
import re


# returns the field as a tuple

def wholeFieldPredicate(field):
    return (field, )


# returns the tokens in the field as a tuple, split on whitespace

def tokenFieldPredicate(field):
    return tuple(field.split())


# Contain common integer

def commonIntegerPredicate(field):
    return tuple(re.findall("\d+", field))


def nearIntegersPredicate(field):
    ints = sorted([int(i) for i in re.findall("\d+", field)])
    return tuple([(i - 1, i, i + 1) for i in ints])


def commonFourGram(field):
    return tuple([field[pos:pos + 4] for pos in xrange(0, len(field),
                 4)])


def commonSixGram(field):
    return tuple([field[pos:pos + 6] for pos in xrange(0, len(field),
                 6)])


def sameThreeCharStartPredicate(field):
    if len(field) < 3:
        return ()

    return (field[:3], )


def sameFiveCharStartPredicate(field):
    if len(field) < 5:
        return ()

    return (field[:5], )


def sameSevenCharStartPredicate(field):
    if len(field) < 7:
        return ()

    return (field[:7], )


