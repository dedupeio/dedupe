## patent_util.py
## Author: Mark Huberty
## Provides utility functions for the patent_example

import re
import random
import examples.shared.AsciiDammit as AsciiDammit
import csv
import collections
import dedupe
import numpy as np
import pandas as pd
import collections
import time
import operator
from dedupe.distance.affinegap import normalizedAffineGapDistance

def Levenshtein(s1, s2) :
    return normalizedAffineGapDistance(s1, s2, 
                                       matchWeight = 0, 
                                       mismatchWeight = 1, 
                                       gapWeight = 1, 
                                       spaceWeight = 1, 
                                       abbreviation_scale = 0.125)

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of
    [AsciiDammit](https://github.com/tnajdek/ASCII--Dammit) and
    Regex. Things like casing, extra spaces, quotes and new lines can
    be ignored.
    """

    column = AsciiDammit.asciiDammit(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    return column



def readDataFrame(df, set_delim='**'):
    """
    Read in our data from a pandas DataFrame as an in-memory database.
    Reformat the data into a dictionary of records,
    where the key is a unique record ID and each value is a 
    [frozendict](http://code.activestate.com/recipes/414283-frozen-dictionaries/) 
    (hashable dictionary) of the row fields.

    Remap columns for the following cases:
    - Lat and Long are mapped into a single LatLong tuple
    - Class and Coauthor are stored as delimited strings but mapped into sets

    **Currently, dedupe depends upon records' unique ids being integers
    with no integers skipped. The smallest valued unique id must be 0 or
    1. Expect this requirement will likely be relaxed in the future.**
    """

    data_d = {}

    for idx, dfrow in df.iterrows():
        # print type(dfrow)
        row_out = {}
        classes = dfrow['Class'].split(set_delim)
        coauthors = dfrow['Coauthor'].split(set_delim)
        classes = [preProcess(c) if isinstance(c, str) else '' for c in classes]
        coauthors = [preProcess(c) if isinstance(c, str) else '' for c in coauthors]
        if isinstance(dfrow['Name'], str):
            name = preProcess(dfrow['Name'])
        else:
            name = ''
        row_out['Class'] = frozenset(classes)
        row_out['Coauthor'] = frozenset(coauthors)
        # row_out['Coauthor_Count'] = len(coauthors)
        # row_out['Class_Count'] = len(classes)
        row_out['LatLong'] = (float(dfrow['Lat']), float(dfrow['Lng']))
        row_out['Name'] = name
        row_tuple = [(k, v) for (k, v) in row_out.items()]
        data_d[idx] = dedupe.core.frozendict(row_tuple)
            
    return data_d

def compute_block_summary(blocks):
    """
    Given a block map as returned from return_block_map,
    compute some summary statistics
    """
    block_count = len(blocks)
    block_len = [len(block) for block in blocks]
    max_block_len = np.max(block_len)
    median_block_len = np.median(block_len)
    mean_block_len = np.mean(block_len)
    print 'Number of blocks: %s' % block_count
    print 'Maximum block length: %s' % max_block_len
    print 'Median block length: %s' % median_block_len
    print 'Mean block length %s' % mean_block_len
    return 0





# Consolidate functions
# These are used for the two-stage disambiguation process
# Each takes a pandas Series object and returns a scalar 
def consolidate_unique(x):
    """
    Returns the first value in the series
    """
    return x.values[0]

def consolidate_geo(x):
    """
    For x as a lat or long series, returns the
    most frequent non-zero value
    """
    geo_counts = {}
    for g in x.values:
        g = float(g)
        if g != 0.0:
            if g in geo_counts:
                geo_counts[g] += 1
            else:
                geo_counts[g] = 1
    if len(geo_counts) > 0:
        sorted_geo = sorted(geo_counts.iteritems(),
                            key=operator.itemgetter(1),
                            reverse=True
                            )
        return sorted_geo[0][0]
    else:
        return 0.0

def consolidate_set(x, delim='**', maxlen=100):
    """
    Consolidates all multi-valued strings in x
    into a unique set of maximum length maxlen.

    Returns a multivalued string separated by delim
    """
    vals = [v.split(delim) for v in x.values if isinstance(v, str)]
    val_set = [v for vset in vals for v in vset]
    val_set = list(set(val_set))
    if len(val_set) > 0:
        if len(val_set) > maxlen:
            rand_idx = random.sample(range(len(val_set)), maxlen)
            val_set = [val_set[idx] for idx in rand_idx]
        out = delim.join(val_set)
    else:
        out = ''
    return out

def consolidate(df, key, agg_dict):
    """
    Aggregates a pandas dataframe into a set of unique rows specified by
    the key. Aggregation method should be specified by column in agg_dict,
    which takes a column_name:function format.

    Used in the _twostage method to consolidate the data based on the
    unique record IDs assigned from the first stage of disambiguation
    """
    grouped = df.groupby(key)
    records = grouped.agg(agg_dict)
    return records

def blockingSettingsWrapper(ppc, uncovered_dupes, dedupe_instance, maxtries=10):
    """
    Wrapper around the dedupe blocker to tun the ppc / uncovered values in case
    no valid blocking can be found. Assumes that the ppc is set too tightly, or the uncovered_dupes
    too broadly, for effect.
    """
    blockerError = True
    try_count = 0
    while blockerError:
        if try_count > maxtries or uncovered_dupes < 1 or ppc > 0.5:
            blocker = None
            break
        try: 
            blocker = dedupe_instance.blockingFunction(ppc, uncovered_dupes)
            blockerError = False
        except ValueError:
            ppc += ppc / 2
            uncovered_dupes -= 1
            try_count += 1
        
    return (blocker, ppc, uncovered_dupes)


def subset_nth_quantile(g, n):
    """
    given a grouped object, returns the indices corresponding
    to the largest n group sizes
    """
    g_size = g.size()
    g_size.sort()
    idx_out = g_size.index[-n:]
    g_tot = np.sum(g_size)
    g_sub = np.sum(g_size.ix[idx_out])
    print 'ratio = %f' % (float(g_sub) / g_tot)
    return idx_out

def find_potential_matches(s1, s2, threshold):
    """
    Given comparable fields in two DataFrames, returns
    the indices of df2 whose field is within threshold of the field
    in df1
    """
    df2_idx = {}
    for count, s in enumerate(s1.drop_duplicates()):
        if count > 0:
            print len(df2_idx)
            iter_time = time.time() - start_time
            print 'Computed inventor matches in %f seconds' % np.round(iter_time, 3)

        start_time = time.time()
        print 'Matching top inventor %s' % count
        for jdx in s2.index:
            if jdx not in df2_idx:
                l = Levenshtein(s, s2.ix[jdx])
                if l >=threshold:
                    df2_idx[jdx] = 1
    return df2_idx.keys()


