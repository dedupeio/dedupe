from collections import deque
import random
import itertools
import warnings

def dedupeBlockedSample(sample_size, predicates, data) :
    items = data.items()
    random.shuffle(items)
    items = deque(items)
    
    blocked_sample = []
    remaining_sample = sample_size - len(blocked_sample)

    while remaining_sample :
        new_sample = list(samplePredicates(remaining_sample, 
                                           predicates,
                                           items))
        
        blocked_sample.extend(itertools.chain.from_iterable(new_sample))
        
        predicates = list(itertools.compress(predicates, new_sample))

        remaining_sample = sample_size - len(blocked_sample)
        

    return blocked_sample

def samplePredicates(sample_size, predicates, items) :

    subsample_counts = evenSplits(sample_size, len(predicates))

    requested_samples = [(count, predicate) 
                         for count, predicate
                         in zip(subsample_counts, predicates)
                         if count]

    n_items = len(items)

    for subsample_size, predicate in requested_samples :

        items.rotate(random.randrange(n_items))

        yield samplePredicate(subsample_size,
                              predicate,
                              items)

def samplePredicate(subsample_size, predicate, items) :

    sample = []
    block_dict = {}
    predicate_function = predicate.func
    field = predicate.field

    for pivot, (index, record) in enumerate(items) :
        if pivot == 10000:
            if len(block_dict) + len(sample) < 10 :
                return sample

        block_keys = predicate_function(record[field])
        
        for block_key in block_keys:
            if block_key not in block_dict :
                block_dict[block_key] = index
            else :
                sample.append((block_dict[block_key],
                               index))
                subsample_size -= 1
                del block_dict[block_key]

                if subsample_size :
                    break
                else :
                    return sample

    else :
        return sample

def evenSplits(total_size, num_splits) :
    avg = total_size/float(num_splits) 
    split = 0
    for _ in xrange(num_splits) :
        split += avg - int(split)
        yield int(split)

