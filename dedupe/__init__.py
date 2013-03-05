#!/usr/bin/python
# -*- coding: utf-8 -*-
__all__ = ['affinegap',
           'blocking',
           'clustering',
           'core',
           'lr',
           'predicates',
           'training_sample',
           'crossvalidation',
           'dedupe',
           'jaccard',
           'haversine'
           ]

import affinegap
import haversine
import jaccard
import blocking
import clustering
import core
import lr
import predicates
import training
import crossvalidation
from api import Dedupe
from core import randomPairs
from convenience import dataSample
from convenience import blockData
