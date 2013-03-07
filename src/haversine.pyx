#!python
#cython: boundscheck=False, wraparound=False

from libc cimport limits
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, asin, sqrt

# cdef extern from 'math.h':
#     double sin(double x)
#     double cos(double x)
#     double asin(double x)
#     double sqrt(double x)
    
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

    cdef float a = sin(dlat/2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon/2)**2

    cdef float c = 2 * asin(sqrt(a))
    cdef float km = 6371 * c
    return km

def split_latlong(latlong, delim):
    lat, lng = latlong.split(delim)
    return float(lat), float(lng)

cpdef float compareLatLong(char *latlong1, char *latlong2, delim='**'):
    lat1, long1 = split_latlong(latlong1, delim)
    lat2, long2 = split_latlong(latlong2, delim)
    dist = haversine(long1, lat1, long2, lat2)
    return dist
