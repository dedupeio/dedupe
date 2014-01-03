#ifndef CUTILS_H
#define CUTILS_H

#include <stdio.h>
#include <ext/hash_map>
#include <ext/hash_set>
#include <iostream>
#include <math.h>
#include <queue>
#include <utility>

#include <new>
#include <Python.h>

// void* operator new (size_t size);
// void operator delete (void* p);
  
                    
typedef std::pair<int, double> AVPair;
typedef std::vector<AVPair> AVPairs;

typedef __gnu_cxx::hash_map<int, double> IntDoubleMap;

typedef std::vector<void *> CAVArray;
void del_store(CAVArray* s);
void add_store(CAVArray* s, void *av);

typedef int& _intref;

void set_dict(IntDoubleMap* dict, int a, double v);
double get_dot(IntDoubleMap* d1, IntDoubleMap* d2);
double cosine_dict(IntDoubleMap *d);
void normalize_dict(IntDoubleMap *d);

char* make_str(IntDoubleMap* dict);

void merge_vectors(IntDoubleMap *v1, IntDoubleMap *v2);

IntDoubleMap* av_fromstring(char *);

// struct PyObject_Hash_Class {
//     long operator()(PyObject* a) const {
//         return PyObject_Hash(a);
//     }
// };

// KNN Neighbors 
struct VoidPtr_Hash_Class {
    long operator()(void* a) const {
        return (long) a;
    }
};

typedef __gnu_cxx::hash_set<void*, VoidPtr_Hash_Class> NeighborSet;
void add_neighbor_set(NeighborSet* ns, void* k);
int contains_ns (NeighborSet* ns, void *ele);

typedef std::pair<void*,double> NeighborPair;
typedef std::vector<NeighborPair> NeighborVector;

struct NeighborPair_comp {
    bool operator()(const NeighborPair& a, const NeighborPair& b){
        return a.second > b.second;
    }
};

void add_neighbor(NeighborVector* nv, void* neighbor, double score);
void partial_sort_nv(NeighborVector* nv, int K);

// Inverted Indexing
typedef __gnu_cxx::hash_map<int, void*> AtomToAtomVectorStoreMap;
int has_a2avs(AtomToAtomVectorStoreMap* dict, int a);
void set_a2avs(AtomToAtomVectorStoreMap* dict, int a, void* v);
void del_a2avs(AtomToAtomVectorStoreMap* dict);


// General purpose
typedef std::vector<int> IntVector;

#endif

