from collections import deque
import random
import itertools
import warnings
from collections import defaultdict

def dedupeBlockedSample(sample_size, predicates, data) :
    items = data.items()
    
    blocked_sample = set([])
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and predicates :
        random.shuffle(items)
        random.shuffle(predicates)

        new_sample = list(dedupeSamplePredicates(remaining_sample, 
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


        
    return blocked_sample

def dedupeSamplePredicates(sample_size, predicates, items) :

    subsample_counts = evenSplits(sample_size, len(predicates))

    requested_samples = [(count, predicate) 
                         for count, predicate
                         in zip(subsample_counts, predicates)
                         if count]

    n_items = len(items)

    for subsample_size, predicate in requested_samples :

        items.rotate(random.randrange(n_items))

        yield dedupeSamplePredicate(subsample_size,
                              predicate,
                              items)

def dedupeSamplePredicate(subsample_size, predicate, items) :

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


def linkBlockedSample(sample_size, predicates, d1, d2) :

    items1 = d1.items()
    items2 = d2.items()

    blocked_sample = set([])
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and predicates:
        random.shuffle(items1)
        random.shuffle(items2)
        random.shuffle(predicates)

        new_sample = list(
            linkSamplePredicates(   remaining_sample,
                                    predicates, 
                                    deque(items1), 
                                    deque(items2)      ) 
            )

        blocked_sample.update(itertools.chain.from_iterable(new_sample))

        growth = len(blocked_sample) - previous_sample_size
        growth_rate = growth/float(remaining_sample)

        remaining_sample = sample_size - len(blocked_sample)
        previous_sample_size = len(blocked_sample)

        if growth_rate < 0.001 :
            warnings.warn("%s blocked samples were requested, but only able to sample %s" % 
                (sample_size, len(blocked_sample)))
            break

        predicates = [pred for pred, subsample in zip(predicates, new_sample) if subsample]

    return blocked_sample


def linkSamplePredicates(sample_size, predicates, items1, items2) :

    subsample_counts = evenSplits(sample_size, len(predicates))

    requested_samples = [   (count, predicate)
                            for count, predicate
                            in zip(subsample_counts, predicates)
                            if count     ]
    
    n_1 = len(items1)
    n_2 = len(items2)

    for subsample_size, predicate in requested_samples:

        items1.rotate(random.randrange(n_1))
        items2.rotate(random.randrange(n_2))

        yield linkSamplePredicate(subsample_size, predicate, items1, items2)


def linkSamplePredicate(subsample_size, predicate, items1, items2) :
    pairs = []
    block_dict_1 = defaultdict(list)
    block_dict_2 = defaultdict(list)
    predicate_function = predicate.func
    field = predicate.field

    larger_len = max(len(items1), len(items2))
    smaller_len = min(len(items1), len(items2))

    for i in range(smaller_len):
        if i == 1000:
            if len(pairs) <1 :
                return pairs
        if i == 10000:
            if len(pairs) <10 :
                return pairs
        block_keys_1 = predicate_function( items1[i][1][field] )
        block_keys_2 = predicate_function( items2[i][1][field] )
        for block_key in block_keys_1:
            if subsample_size == 0:
                return pairs
            block_dict_1[block_key].append(items1[i][0])
            if len(block_dict_2[block_key]) > 0:
                pairs.append(( block_dict_1[block_key].pop(), block_dict_2[block_key].pop() ))
                subsample_size = subsample_size - 1
        for block_key in block_keys_2:
            if subsample_size == 0:
                return pairs
            block_dict_2[block_key].append(items2[i][0])
            if len(block_dict_1[block_key]) > 0:
                pairs.append(( block_dict_1[block_key].pop(), block_dict_2[block_key].pop() ))
                subsample_size = subsample_size - 1
    """
    if len(items1) > len(items2):
        for i in range(smaller_len, larger_len):
            if i == 10000:
                if len(pairs) <10 :
                    return pairs
            if subsample_size == 0:
                return pairs
            block_keys = predicate_function( items1[i][1][field] )
            for block_key in block_keys:
                if len(block_dict_2[block_key]) > 0:
                    pairs.append(( items1[i][0], block_dict_2[block_key].pop() ))
                    subsample_size = subsample_size - 1
    elif len(items2) > len(items1):
        for i in range(smaller_len, larger_len):
            if i == 10000:
                if len(pairs) <10 :
                    return pairs
            if subsample_size == 0:
                return pairs
            block_keys = predicate_function( items2[i][1][field] )
            for block_key in block_keys:
                if len(block_dict_1[block_key]) > 0:
                    pairs.append(( block_dict_1[block_key].pop(), items2[i][0] ))
                    subsample_size = subsample_size - 1
    """
    
    return pairs

