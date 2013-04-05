import re
import random
import AsciiDammit
import csv
import collections
import dedupe
import numpy as np
import pandas as pd
import collections
import time
import operator


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


def readData(filename, set_delim='**'):
    """
    Read in our data from a CSV file and create a dictionary of records, 
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
    with open(filename) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # for k in row:
            #     row[k] = preProcess(row[k])
            row['LatLong'] = (float(row['Lat']), float(row['Lng']))
            del row['Lat']
            del row['Lng']
            row['Class'] = frozenset(row['Class'].split(set_delim))
            row['Coauthor'] = frozenset([author for author
                                         in row['Coauthor'].split(set_delim)
                                         if author != 'none'])
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            
            data_d[idx] = dedupe.core.frozendict(clean_row)
            
    return data_d


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

    Assumes the index in the dataframe is already set to a continuous-integer
    unique row ID.
    
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
        row_out['LatLong'] = (float(dfrow['Lat']), float(dfrow['Lng']))
        row_out['Name'] = name
        row_tuple = [(k, v) for (k, v) in row_out.items()]
        data_d[idx] = dedupe.core.frozendict(row_tuple)
            
    return data_d

def return_block_map(d, b):
    """
    For data d and a blocker b, return a
    block:[record_id] map as a dict
    """
    block_map = collections.defaultdict(list)
    for record_id, record in d.iteritems():
        for block_id in b((record_id, record)):
            block_map[block_id].append(record_id)

    print 'Blocking done'
    compute_block_summary(block_map)
    return block_map


def compute_block_summary(block):
    """
    Given a block map as returned from return_block_map,
    compute some summary statistics
    """
    block_count = len(block)
    block_len = [len(v) for k,v in block.iteritems()]
    max_block_len = np.max(block_len)
    median_block_len = np.median(block_len)
    print 'Number of blocks: %s' % block_count
    print 'Maximum block length: %s' % max_block_len
    print 'Median block length: %s' % median_block_len
    return 0


def return_threshold_data(block_map, d, n_samples=1000):
    """
    Given a block map and a corresponding data object, return
    n_samples random blocks as a list of tuples of form
    (record_id, record)
    """
    subset = random.sample(range(len(block_map.keys())), n_samples)
    threshold_data_ids = [block_map[block_map.keys()[i]] for i in subset]
    threshold_data = []
    for id_list in threshold_data_ids:
        record_list = [(id, d[id]) for id in id_list]
        threshold_data.append(tuple(record_list))
    return tuple(threshold_data)


def candidates_gen(block_map, block_keys, d) :
    """
    Builds a record generator by block ID for deduping.
    """
    start_time = time.time()
    for i, block_key in enumerate(block_keys):
        if i % 1000 == 0 :
            print i, "blocks"
            print time.time() - start_time, "seconds"
            if i > 0:
                print (time.time() - start_time) / i, "seconds per block"
            
        yield ((id, d[id]) for id in block_map[block_key])

def consolidate_deduped_data():
    """
    Given results of a de-duping, consolidates to a
    unique record set. 
    """


def check_convergence(orig_record_count, record_count, convergence_threshold):
    """
    Test for convergence, defined as less than threshold change
    in record count
    """
    record_ratio = float(record_count) / orig_record_count
    if record_ratio < convergence_threshold:
        return True
    else:
        return False

    
def consolidate_unique(x):
    return x.values[0]

def consolidate_geo(x):
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
    grouped = df.groupby(key)
    records = grouped.agg(agg_dict)
    return records
