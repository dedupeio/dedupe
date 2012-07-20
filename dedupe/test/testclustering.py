from clustering import *


dupes =       (((1,2), .95),
                ((1,3), .7),
                ((1,4), .2),
                ((2,5), .6),
                ((2,7), .8),
                ((2,3), .9),
                ((3,5), .2)
                )

print cluster(dupes,
              sparseness_threshold = 4,
              k_nearest_neighbors = 6,
              neighborhood_multiplier = 2,
              estimated_dupe_fraction = 1.0) 
