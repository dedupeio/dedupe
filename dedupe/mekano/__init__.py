"""Everything to do with representing documents as numbers (atoms).

   See the following for more details:
     - L{atomvector} provides the core functionality of representing documents as sparse vectors (L{AtomVector} class).
     - L{atomfactory} maintains the mapping between textual tokens and numbers (atoms).
     - L{invidx} provides functionality for creating inverted indexes.
     - L{weightvectors} provides functionality for created weighted document vectors.
     - L{corpusstats} maintains various corpus statistics (like term frequencies) to support the creation of weighted vectors.

"""

from atomfactory import AtomFactory
from atomvector import AtomVector
from atomvectorstore import AtomVectorStore
from invidx import InvertedIndex
from weightvectors import WeightVectors
from corpusstats import CorpusStats
