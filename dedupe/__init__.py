#!/usr/bin/python
# -*- coding: utf-8 -*-
__all__ = ['affinegap',
           'blocking',
           'clustering',
           'core',
           'lr',
           'predicates',
           'crossvalidation',
           'distance'
           ]

#from distance import affinegap
from distance import affinegap
import distance 
import blocking
import clustering
import core
import lr
import predicates
import training
import crossvalidation
import datamodel
import backport
from api import StaticDedupe, Dedupe
from api import StaticRecordLink, RecordLink
from core import randomPairs
from convenience import consoleLabel, trainingDataDedupe, trainingDataLink
from AsciiDammit import asciiDammit
import backport
