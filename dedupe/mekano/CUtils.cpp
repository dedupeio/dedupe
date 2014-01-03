#include <CUtils.h>

using namespace std;
using namespace __gnu_cxx;

// void* operator new (size_t size) {
//   return PyMem_Malloc(size);
// }

// void operator delete (void* p) {
//   PyMem_Free(p);
// }


void set_dict(IntDoubleMap* dict, int a, double v) {
  IntDoubleMap::iterator it;
  if (v != 0) {
    (*dict)[a] = v;
  } else {
    it = dict->find(a);
    if (it != dict->end())
      dict->erase(it);
  }
}

double get_dot(IntDoubleMap* d1, IntDoubleMap* d2) {
  int a;
  IntDoubleMap::const_iterator i = d1->begin();
  IntDoubleMap::const_iterator o_end = d2->end();
  double ret = 0.0;
  int c;
  while (1) {
    c = (i != d1->end());
    if (!c) break;
    a = i->first;
    IntDoubleMap::const_iterator j = d2->find(a);
    if (j != o_end)
      ret += i->second * j->second;
    i++;
  }
  return ret;
}

double cosine_dict(IntDoubleMap *d) {
  IntDoubleMap::const_iterator i = d->begin();
  double ret = 0.0;
  while (i != d->end()) {
    ret += i->second*i->second;
    i++;
  }
  return sqrt(ret);
}

void normalize_dict(IntDoubleMap *d) {
  double cos = cosine_dict(d);
  IntDoubleMap::iterator i = d->begin();
  while (i != d->end()) {
    i->second /= cos;
    i++;
  }
}

void merge_vectors(IntDoubleMap *v1, IntDoubleMap *v2) {
  IntDoubleMap::const_iterator i2, end2;
  i2 = v2->begin();
  end2 = v2->end();
  while (i2 != end2) {
    (*v1)[i2->first] += i2->second;
    i2++;
  }
}
  

char* make_str(IntDoubleMap* dict) {
  int l = dict->size();
  int pos = 0;
  char* buf = new char[18*l];
  IntDoubleMap::const_iterator it = dict->begin();
  while (it != dict->end()) {
    int r = sprintf(buf+pos, "%d:%-7.4f ", it->first, it->second);
    pos += r;
    it++;
  }
  if (pos > 0) {
    buf[pos-1] = 0;
  }
  return buf;
}

IntDoubleMap* av_fromstring(char *p) {
  IntDoubleMap* d = new IntDoubleMap();
  int a; double v;
  while(*p) {
    int m = sscanf(p, "%d:%lf ", &a, &v);
    if (m < 2) return d;
    (*d)[a] = v;
    do {
      p++;
      if (*p == 0) return d;
      if (*p == ' ') break;
    } while(1);
    p++;
  }
  return d;
}


// AVPairs* AtomVector::items() {
//   AVPairs *ret = new AVPairs;
// //  cout << "atomvector.cpp address=" << (void *) ret << "\n";
//   IntDoubleMap::const_iterator i = vec.begin();
//   IntDoubleMap::const_iterator end = vec.end();
//   priority_queue<int> Q;
//   while (i != end) {
//     Q.push(-(i->first));
//     i++;
//   }
//   while (!Q.empty()) {
//     int a = -Q.top();
//     Q.pop();
//     ret->push_back(AVPair(a, vec[a]));
//   }
//   return ret;
// }

    
int contains_ns (NeighborSet* ns, void *ele) {
    if (ns->find(ele) == ns->end()) return 0;
    else return 1;
}

void add_neighbor(NeighborVector* nv, void* neighbor, double score) {
    nv->push_back(NeighborPair(neighbor, score));
}

void partial_sort_nv(NeighborVector* nv, int K) {
    int s = nv->size();
    if (K > s) { K = s; }
    partial_sort(nv->begin(), nv->begin()+K, nv->end(), NeighborPair_comp());
}

int has_a2avs(AtomToAtomVectorStoreMap* dict, int a) {
    AtomToAtomVectorStoreMap::const_iterator it = dict->find(a);
    if (it == dict->end()) return 0;
    else return 1;
}

void set_a2avs(AtomToAtomVectorStoreMap* dict, int a, void* v) {
    Py_INCREF((PyObject*) v);
    (*dict)[a] = v;
}

void del_a2avs(AtomToAtomVectorStoreMap* dict) {
    AtomToAtomVectorStoreMap::const_iterator it= dict->begin();
    AtomToAtomVectorStoreMap::const_iterator end= dict->end();
    while (it != end) {
        Py_DECREF(it->second);
        it++;
    }
    delete dict;
}

void del_store(CAVArray* s) {
    CAVArray::const_iterator it = s->begin();
    CAVArray::const_iterator end = s->end();
    while (it != end) {
        Py_DECREF(*it);
        it++;
    }
    delete s;
}

void add_store(CAVArray* s, void *av) {
    Py_INCREF((PyObject*) av);
    s->push_back(av);
}