#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from libc.stdlib cimport malloc, free
import math

## Equivalent to 3.1415927 / 180
cdef float PI_RATIO = 0.017453293

cpdef float deg2rad(float deg):
    cdef float rad = deg * PI_RATIO
    return rad
    
cpdef float haversine(float lon1, float lat1, float lon2, float lat2):
    cdef float rlon1 = deg2rad(lon1)
    cdef float rlon2 = deg2rad(lon2)
    cdef float rlat1 = deg2rad(lat1)
    cdef float rlat2 = deg2rad(lat2)
    
    cdef float dlon = rlon2 - rlon1
    cdef float dlat = rlat2 - rlat1

    cdef float a = math.sin(dlat/2)**2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon/2)**2

    cdef float c = 2 * math.asin(math.sqrt(a))
    # cdef float c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    cdef float km = 6371 * c
    return km

cpdef split_string(char *valstr, char *delim):
    lat, lng = valstr.split(delim)
    return float(lat), float(lng)

cpdef float compareLatLong(char *latlong1, char *latlong2, delim='**'):
    lat1, long1 = split_string(latlong1, delim)
    lat2, long2 = split_string(latlong2, delim)
    dist = haversine(long1, lat1, long2, lat2)
    return dist
