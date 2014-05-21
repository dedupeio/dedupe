#!/usr/bin/python
# -*- coding: utf-8 -*-
__all__ = ['affinegap',
           'blocking',
           'clustering',
           'core',
           'lr',
           'backport',
           'predicates',
           'crossvalidation',
           'distance'
           ]

#from distance import affinegap
from api import StaticDedupe, Dedupe
from api import StaticRecordLink, RecordLink
from core import randomPairs
from convenience import consoleLabel, trainingDataDedupe, trainingDataLink
from AsciiDammit import asciiDammit
