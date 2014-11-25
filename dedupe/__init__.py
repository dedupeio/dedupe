#!/usr/bin/python
# -*- coding: utf-8 -*-
__all__ = ['blocking',
           'clustering',
           'core',
           'backport',
           'predicates',
           'crossvalidation',
           'distance',
           'centroid'
           ]

#from distance import affinegap
from api import StaticDedupe, Dedupe
from api import StaticRecordLink, RecordLink
from api import StaticGazetteer, Gazetteer
from core import randomPairs, randomPairsMatch, frozendict
from convenience import consoleLabel, trainingDataDedupe, trainingDataLink, canonicalize
