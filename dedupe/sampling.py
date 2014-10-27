from collections import deque
import random
import itertools
import warnings

def dedupeBlockedSample(sample_size, predicates, data) :
    items = data.items()
    
    blocked_sample = set([])
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and predicates :
        random.shuffle(items)
        random.shuffle(predicates)

        new_sample = list(samplePredicates(remaining_sample, 
                                           predicates,
                                           deque(items)))
        
        blocked_sample.update(itertools.chain.from_iterable(new_sample))

        growth = len(blocked_sample) - previous_sample_size
        growth_rate = growth/float(remaining_sample)

        remaining_sample = sample_size - len(blocked_sample)
        previous_sample_size = len(blocked_sample)

        if growth_rate < 0.001 :
            warnings.warn("%s blocked samples were requested, but only able to sample %s" % 
                          (sample_size, len(blocked_sample)))
            break

        predicates = [pred for pred, subsample in zip(predicates, new_sample)
                      if subsample]


        
    return list(blocked_sample)

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
                other_index = block_dict[block_key]
                if other_index > index :
                    index, other_index = other_index, index
                sample.append((other_index,
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

