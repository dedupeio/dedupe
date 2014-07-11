import numpy

# takes in a list of attribute values for a field,
# evaluates the centroid using the comparator,
# & returns the centroid (i.e. the 'best' value for the field)
def getCentroid( attribute_variants, comparator ):
    n = len(attribute_variants)
    # if all values were empty & ignored in getCanonicalRep, return ''
    dist_matrix = numpy.zeros([n,n])
    # this is a matrix of distances between all strings
    # populate distance matrix by looping through elements of matrix triangle
    for i in range (0,n):
        for j in range (0, i):
            dist = comparator(attribute_variants[i], attribute_variants[j])
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    # find avg distance per string
    avg_dist = dist_matrix.mean(0)
    # find string with min avg distance
    min_dist_indices = numpy.where(avg_dist==avg_dist.min())[0]
    # if there is only one value w/ min avg dist
    if len(min_dist_indices)==1:
        centroid_index = min_dist_indices[0]
        return attribute_variants[centroid_index]
    # if there are multiple values w/ min avg dist
    else:
        return breakCentroidTie( attribute_variants, min_dist_indices )

# finds centroid when there are multiple values w/ min avg distance (e.g. any dupe cluster of 2)
# right now this selects the first among a set of ties, but can be modified to break ties in strings by selecting the longest string
def breakCentroidTie( attribute_variants, min_dist_indices ):
    return attribute_variants[min_dist_indices[0]]

# takes in a cluster of duplicates & data, returns canonical representation of cluster
def getCanonicalRep( dupe_cluster, data_d, data_model):
    canonical_rep = dict()

    #loop through keys & values in data, get centroid for each key
    for key, comparator in data_model.field_comparators.items():
        key_values = []
        for record_id in dupe_cluster :
            #ignore empty values (assume non-empty values always better than empty value for canonical record)
            if data_d[record_id][key] != '':
                key_values.append(data_d[record_id][key])
        if key_values:
            canonical_rep[key] = getCentroid(key_values, comparator)
        else:
            canonical_rep[key] = ''
    return canonical_rep