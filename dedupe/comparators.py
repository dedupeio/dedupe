#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Convenience functions for field-specific comparisons. 
"""

import math

## Haversine distance taken from
## http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km 

def splitLatlong(latlong, delim):
    latlong_split = [float(val) for val in latlong.split(delim)]
    return latlong[0], latlong[1]

def compareLatlong(latlong1, latlong2, delim='**'):
    """
    Assumes a comparison between two latlong fields of form
    lat<delim>long. Returns the haversine distance in km.
    """
    lat1, long1 = split_latlong(latlong1, delim)
    lat2, long2 = split_latlong(latlong2, delim)

    dist = haversine(long1, lat1, long2, lat2)
    return dist


def compareJaccard(class1, class2, delim='**'):
    """
    Assumes a comparison between two class fields, each of the form
    class1<delim>class2<delim>...

    For instance, a field documenting 'fruits someone eats' would be
    banana**apple**orange, etc.

    Returns the jaccard similarity between the fields
    """
    class1 = set(class1.split(delim))
    class2 = set(class2.split(delim))
    numer = len(class1.intersection(class2))
    denom = len(class1.union(class2))

    return numer / float(denom)

