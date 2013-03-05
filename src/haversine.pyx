#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from libc.stdlib cimport malloc, free
import math


cpdef float haversine(float lon1, float lat1, float lon2, float lat2):

    rlon1, rlat1, rlon2, rlat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    cdef float dlon = rlon2 - rlon1
    cdef float dlat = rlat2 - rlat1

    cdef float a = math.sin(dlat/2)**2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon/2)**2

    cdef float c = 2 * math.asin(math.sqrt(a))
    # cdef float c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    cdef float km = 6371 * c
    return km

cpdef split_string(char *valstr, delim):
    out = [float(val) for val in valstr.split(delim)]
    return out[0], out[1]

cpdef float compareLatLong(char *latlong1, char *latlong2, delim='**'):
    lat1, long1 = split_string(latlong1, delim)
    lat2, long2 = split_string(latlong2, delim)
    dist = haversine(long1, lat1, long2, lat2)
    return dist
