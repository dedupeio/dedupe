#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use dedupe with a comma separated values
(CSV) file. All operations are performed in memory, so will run very
quickly on datasets up to ~10,000 rows.

We start with a CSV file containing our messy data. In this example,
it is listings of early childhood education centers in Chicago
compiled from several different sources.

The output will be a CSV with our clustered results.

For larger datasets, see our [mysql_example](http://open-city.github.com/dedupe/doc/mysql_example.html)
"""

import os
import csv
import re
import collections
import logging
import optparse
import time
import sys
import pandas as pd
import patent_util
import math
import AsciiDammit

import dedupe
from dedupe.distance import cosine
sys.modules['cosine'] = cosine

def idf(i, j) :
    i = int(i)
    j = int(j)
    max_i = max([i,j])
    return math.log(len(data_d)/int(max_i))


# ## Logging
# Dedupe uses Python logging to show or suppress verbose output. Added
# for convenience.  To enable verbose logging, run `python
# examples/csv_example/csv_example.py -v`

optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING 
if opts.verbose == 1:
    log_level = logging.INFO
elif opts.verbose >= 2:
    log_level = logging.DEBUG
logging.basicConfig(level=log_level)


# ## Setup

# Switch to our working directory
# Set the input file
# And the output filepaths
os.chdir('./examples/patent_example/')
input_file = '../../../psClean/data/dedupe_input/person_records/dedupe_input_de.csv'
output_file_root = 'patstat_output_de_20April2013_'
settings_file_root = 'patstat_settings_de_20April2013_'
training_file_root = 'patstat_training_de_20April2013_'
patent_file = '../../../psClean/data/dedupe_input/person_patent/de_person_patent_map.csv'



print 'importing data ...'
input_df = pd.read_csv(input_file)
input_df.Class.fillna('', inplace=True)
input_df.Coauthor.fillna('', inplace=True)
input_df.Lat.fillna('0.0', inplace=True)
input_df.Lng.fillna('0.0', inplace=True)
input_df.Name.fillna('', inplace=True)

# input_df = input_df[:30000]

rounds = [1, 2]
recall_weights = [1, 2]
ppcs = [0.0001, 0.001]
dupes = [5, 5]
twostage = [False, True]
#dupes = [10, 5, 1]

## Start the by-round labeling
for idx, r in enumerate(rounds):

    r_twostage = twostage[idx]
    r_recall_wt = recall_weights[idx]
    r_ppc = ppcs[idx]
    r_uncovered_dupes = dupes[idx]

    r_settings_file = settings_file_root + str(r) + '.json'
    r_output_file = output_file_root + str(r) + '.csv'
    r_training_file = training_file_root + str(r) + '.json'

    # If this is the first round, take the native input
    # If the nth round, consolidate data on the nth index
    # and read in the resulting dataframe.
    if idx == 0:
        data_d = patent_util.readDataFrame(input_df)
    else:
        cluster_agg_dict = {'Name': patent_util.consolidate_unique,
                            'Lat': patent_util.consolidate_geo,
                            'Lng': patent_util.consolidate_geo,
                            'Class': patent_util.consolidate_set,
                            'Coauthor': patent_util.consolidate_set
                            }
        #input_file.set_index(cluster_name)
        consolidated_input = patent_util.consolidate(input_df,
                                                     cluster_name,
                                                     cluster_agg_dict
                                                     )
        if r_twostage:
            # Here, first find the top N patenters, then reduce the consolidated
            # data to those patenters, then append likely matches and just dedupe that
            
            df_patent = pd.read_csv(patent_file)
            # Merge in the consolidated data
            invpat = pd.merge(input_df,
                              df_patent,
                              left_on='Person',
                              right_on='Person',
                              how='inner'
                              )
            invpat_grouped = invpat.groupby(cluster_name)
            top_idx = patent_util.subset_nth_quantile(invpat_grouped, 500)
            del invpat, invpat_grouped
            candidate_inputs = consolidated_input.drop(top_idx, axis=0)
            consolidated_input = consolidated_input.ix[top_idx]

            addl_idx = patent_util.find_potential_matches(consolidated_input.Name,
                                                          candidate_inputs.Name,
                                                          0.8
                                                          )

            addl_data = candidate_inputs.ix[addl_idx]
            consolidated_input = pd.concat([consolidated_input,
                                           addl_data],
                                           axis=0
                                           )

            # Reset the index so that it is sequential. Then
            # store the new:old map
            consolidated_input.reset_index(inplace=True)
            index_map = consolidated_input['index'].to_dict()
        
        data_d = patent_util.readDataFrame(consolidated_input)
        del consolidated_input
        input_df.set_index(cluster_name, inplace=True)
        

## Build the comparators
    coauthors = [row['Coauthor'] for cidx, row in data_d.items()]
    classes = [row['Class'] for cidx, row in data_d.items()]
    class_comparator = dedupe.distance.cosine.CosineSimilarity(classes)
    coauthor_comparator = dedupe.distance.cosine.CosineSimilarity(coauthors)

# ## Training
    if os.path.exists(r_settings_file):
        print 'reading from', r_settings_file
        deduper = dedupe.Dedupe(r_settings_file)

    else:
        # To train dedupe, we feed it a random sample of records.
        data_sample = dedupe.dataSample(data_d, 3000000)
          # Define the fields dedupe will pay attention to
        fields = {
            'Name': {'type': 'String', 'Has Missing':True},
            'LatLong': {'type': 'LatLong', 'Has Missing':True},
            'Class': {'type': 'Custom', 'comparator':class_comparator},
            'Coauthor': {'type': 'Custom', 'comparator': coauthor_comparator}# ,
            # 'Class_Count': {'type': 'Custom', 'comparator': idf},
            # 'Coauthor_Count': {'type': 'Custom', 'comparator': idf},
            # 'Class_Count_Class': {'type': 'Interaction',
            #                       'Interaction Fields': ['Class_Count', 'Class']
            #                       },
            # 'Coauthor_Count_Coauthor': {'type': 'Interaction',
            #                             'Interaction Fields': ['Coauthor_Count', 'Coauthor']
            #                             }
            }

        # Create a new deduper object and pass our data model to it.
        deduper = dedupe.Dedupe(fields)

        # If we have training data saved from a previous run of dedupe,
        # look for it an load it in.
        # __Note:__ if you want to train from scratch, delete the training_file
        # The json file is of the form:
        # {0: [[{field:val dict of record 1}, {field:val dict of record 2}], ...(more nonmatch pairs)]
        #  1: [[{field:val dict of record 1}, {field_val dict of record 2}], ...(more match pairs)]
        # }
        if os.path.exists(r_training_file):
            print 'reading labeled examples from ', r_training_file
            deduper.train(data_sample, r_training_file)

        # ## Active learning

        # Starts the training loop. Dedupe will find the next pair of records
        # it is least certain about and ask you to label them as duplicates
        # or not.

        # use 'y', 'n' and 'u' keys to flag duplicates
        # press 'f' when you are finished
        print 'starting active labeling...'
        deduper.train(data_sample, dedupe.training.consoleLabel)

        # When finished, save our training away to disk
        #deduper.writeTraining(r_training_file)

# ## Blocking
    deduper.blocker_types.update({'Custom': (dedupe.predicates.wholeSetPredicate,
                                             dedupe.predicates.commonSetElementPredicate),
                                  'LatLong' : (dedupe.predicates.latLongGridPredicate,)
                                  }
                                 )
    time_start = time.time()
    print 'blocking...'
    # Initialize our blocker, which determines our field weights and blocking 
    # predicates based on our training data
    #blocker = deduper.blockingFunction(r_ppc, r_uncovered_dupes)
    blocker, ppc_final, ucd_final = patent_util.blockingSettingsWrapper(r_ppc,
                                                                        r_uncovered_dupes,
                                                                        deduper
                                                                        )

    if not blocker:
        print 'No valid blocking settings found'
        print 'Starting ppc value: %s' % r_ppc
        print 'Starting uncovered_dupes value: %s' % r_uncovered_dupes
        print 'Ending ppc value: %s' % ppc_final
        print 'Ending uncovered_dupes value: %s' % ucd_final
        break

    time_block_weights = time.time()
    print 'Learned blocking weights in', time_block_weights - time_start, 'seconds'

    # Save our weights and predicates to disk.
    # If the settings file exists, we will skip all the training and learning
    #deduper.writeSettings(r_settings_file)

    # Generate the tfidf canopy as needed
    print 'generating tfidf index'
    full_data = ((k, data_d[k]) for k in data_d)
    blocker.tfIdfBlocks(full_data)
    del full_data

    # Load all the original data in to memory and place
    # them in to blocks. Return only the block_id: unique_id keys

    blocking_map = patent_util.return_block_map(data_d, blocker)

    keys_to_block = [k for k in blocking_map if len(blocking_map[k]) > 1]
    print '# Blocks to be clustered: %s' % len(keys_to_block)
    
    # Save the weights and predicates
    time_block = time.time()
    print 'Blocking rules learned in', time_block - time_block_weights, 'seconds'
    print 'Writing out settings'
    #deduper.writeSettings(r_settings_file)

    # ## Clustering

    # Find the threshold that will maximize a weighted average of our precision and recall. 
    # When we set the recall weight to 1, we are trying to balance recall and precision
    #
    # If we had more data, we would not pass in all the blocked data into
    # this function but a representative sample.
    
    threshold_data = patent_util.return_threshold_data(blocking_map, data_d)

    print 'Computing threshold'
    threshold = deduper.goodThreshold(threshold_data, recall_weight=r_recall_wt)
    del threshold_data

    # `duplicateClusters` will return sets of record IDs that dedupe
    # believes are all referring to the same entity.

                                                    

    print 'clustering...'
    # Loop over each block separately and dedupe

    clustered_dupes = deduper.duplicateClusters(patent_util.candidates_gen(blocking_map,
                                                                           keys_to_block,
                                                                           data_d
                                                                           ),
                                                threshold
                                                ) 

    print '# duplicate sets', len(clustered_dupes)

    # Extract the new cluster membership 
    ccount = 0
    cluster_membership = collections.defaultdict(lambda : 'x')
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        ccount += len(cluster)
        for record_id in cluster:
            if r_twostage:
                record_id = index_map[record_id]
                cluster_membership[record_id] = cluster_id
            else:
                cluster_membership[record_id] = cluster_id

    # Then write it into the data frame as a sequential index for later use
    r_cluster_index = []
    cluster_counter = 0
    clustered_cluster_map = {}
    excluded_cluster_map = {}
    for df_idx in input_df.index:
        if df_idx in cluster_membership:
            orig_cluster = cluster_membership[df_idx]
            if orig_cluster in clustered_cluster_map:
                r_cluster_index.append(clustered_cluster_map[orig_cluster])
            else:
                clustered_cluster_map[orig_cluster] = cluster_counter
                r_cluster_index.append(cluster_counter)
                cluster_counter += 1
                # print cluster_counter
        else:
            if df_idx in excluded_cluster_map:
                r_cluster_index.append(excluded_cluster_map[df_idx])
            else:
                excluded_cluster_map[df_idx] = cluster_counter
                r_cluster_index.append(cluster_counter)
                cluster_counter += 1

    cluster_name = 'cluster_id_r' + str(r)
    input_df[cluster_name] = r_cluster_index

    # Write out the data frame
    input_df.to_csv(r_output_file)

    # Then reindex and consolidate
    if idx > 0:
        input_df.reset_index(inplace=True)

    print 'Round %s completed' % r
    # END DEDUPE LOOP

print 'Dedupe complete, ran in ', time.time() - start_time, 'seconds'
