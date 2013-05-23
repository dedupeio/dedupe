# In[36]:
import numpy as np
import pandas as pd
import sys

# Pull in the targets
inputs = [i for idx, i in enumerate(sys.argv) if idx > 0]
deduped_output = inputs[0]
reference_data = inputs[1]

cluster_ids = ['cluster_id_r1', 'cluster_id_r2']

for cluster_id in cluster_ids:
    try:
        df_output = pd.read_csv(deduped_output)
        df_reference = pd.read_csv(reference_data)
    except:
        break

    df_map = pd.merge(df_output,
                      df_reference,
                      left_on='Person',
                      right_on='person_id'
                      )
    df_map = df_map[[cluster_id, 'leuven_id', 'person_id']]
    grouped = df_map.groupby(['leuven_id', cluster_id])

    # Count the number of raw person_ids attached to each
    # ref_id:dedupe_id pair.
    pair_ct = grouped.size()

    # There could be a 1:many ref:dedupe ID map. Define precision and
    #  recall as follows:

    # Recall: given the map from ref_id:dedupe_id:raw_id, return the
    #         maximum number of raw_ids assigned to a single dedupe_id for
    #         a single ref_id. Do for all ref_ids
    # Precision: given the map, return the maximum number of of raw_ids
    #            assigned to a single ref_id for a single dedupe_id. Do for all dedupe_ids.
    
    # compute the recall value as <most person ids assigned to a single
    # unique ID also assigned to a single refernce ID
    ref_ct = pair_ct.groupby(level=0)
    max_vals = ref_ct.agg(np.max)
    overall_recall = np.sum(max_vals) / float(np.sum(pair_ct))

    # compute precision as 
    dedupe_ct = pair_ct.groupby(level=1)
    max_vals = dedupe_ct.agg(np.max)
    overall_precision = float(np.sum(max_vals)) / np.sum(pair_ct)

    print 'Cluster level: %s' % cluster_id
    print 'Recall: %f' % overall_recall
    print 'Precision: %f' % overall_precision
