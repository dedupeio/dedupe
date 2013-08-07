"""Classes and functions for creating and managing unique atoms.

The main exposed class is L{AtomFactory}.

Useful functions: L{convertAtom} and L{convertAtomVector}.
"""

from __future__ import with_statement
import cPickle
from atomvector import AtomVector

class AtomFactory:
    """
    A single AtomFactory makes unique atoms for the given
    objects. By atoms, we just mean numbers.
    Objects just have to be hashable.

        >>> af = AtomFactory("mytokens")
        >>> a1 = af["apples"]
        >>> a2 = af["oranges"]
        >>> assert(a1 == 1)
        >>> assert(a2 == 2)
        >>> assert(af(1) == "apples")
        >>> a.lock()                        # Do not allow changes.
        
    Loading/saving:
        >>> a = AtomFactory.fromfile(filename)
        >>> a.save(filename)
    
    @note:  C{af(1)} is candy for C{af.get_object(1)}
    
    """

    def __init__(self, name = "noname"):
        self.name = name
        # make a bi-map
        self.obj_to_atom = {}
        self.atom_to_obj = []
        self.locked = False

    def __repr__(self):
        return "<AtomFactory: %s      %d atoms>" % (self.name, len(self.atom_to_obj))

    def __getitem__(self, obj):
        try:
            return self.obj_to_atom[obj]
        except KeyError:
            if self.locked:
                raise
            a = len(self.atom_to_obj) + 1
            self.obj_to_atom[obj] = a
            self.atom_to_obj.append(obj)
            return a

    def get_object(self, a):
        return self.atom_to_obj[a-1]
    
    def __call__(self, a):
        return self.atom_to_obj[a-1]

    def __len__(self):
        return len(self.atom_to_obj)
    
    def __contains__(self, obj):
        return obj in self.obj_to_atom

    def lock(self):
        """Lock the AtomFactory. 
        
        No new atoms can be added; Only old ones can be retrieved.
        """
        self.locked = True
    
    def remove(self, objects):
        """Returns a new AtomFactory with the given objects removed.
        """
        objects = set(objects)
        new_af = AtomFactory(self.name)
        for obj in self.atom_to_obj:
            if obj not in objects:
                new_af[obj]
        return new_af

    def save(self, filename):
        with open(filename, "w") as fout:
            cPickle.dump(self, fout, -1)
    
    def savetxt(self, filename):
        """Save each object on a line.
        
        This should be enough to reconstruct the AtomFactory,
        and is also useful for things like LDA's vocabulary file.
        """
        with open(filename, "w") as fout:
            for obj in self.atom_to_obj:
                fout.write("%s\n" % obj)
                
    @staticmethod
    def fromfile(filename):
        with open(filename, "r") as fin:
            a = cPickle.load(fin)
        return a

def convertAtom(oldAF, newAF, atom):
    """Convert an atom from one AtomFactory to another.
    
    @param oldAF            : The old AtomFactory to which atom belongs
    @param newAF            : The new AtomFactory
    @param atom             : The atom to convert
    @return                 : The converted atom
    @raise Exception        : If atom cannot be found in oldAF
    """
    o = oldAF.get_object(atom)
    if o not in newAF:
        raise Exception, "%r not in newAF" % o
    return newAF[o]

def convertAtomVector(oldAF, newAF, av):
    """Convert an L{AtomVector} from one AtomFactory to another.
    
    @param oldAF            : The old AtomFactory to which AtomVector av belongs
    @param newAF            : The new AtomFactory
    @param av               : The AtomVector to convert
    @return                 : The converted AtomVector
    """
    new_av = AtomVector(av.name)
    for a, v in av.iteritems():
        try:
            a = convertAtom(oldAF, newAF, a)
            new_av[a] = v
        except Exception:       # todo: why are we suppressing the exception ?!
            pass
    return new_av
