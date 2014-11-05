from collections import deque
import random
import functools
import itertools
import warnings
from collections import defaultdict

def blockedSample(sampler, sample_size, predicates, *args) :
    
    blocked_sample = set([])
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and predicates :
        random.shuffle(predicates)

        new_sample = list(sampler(remaining_sample, 
                                  predicates,
                                  *args))

        filtered_sample = (subsample for subsample 
                           in new_sample if subsample)

        blocked_sample.update(itertools.chain.from_iterable(filtered_sample))

        growth = len(blocked_sample) - previous_sample_size
        growth_rate = growth/float(remaining_sample)

        remaining_sample = sample_size - len(blocked_sample)
        previous_sample_size = len(blocked_sample)

        if growth_rate < 0.001 :
            warnings.warn("%s blocked samples were requested, "
                          "but only able to sample %s"
                          % (sample_size, len(blocked_sample)))
            break

            
        predicates = [pred for pred, pred_sample 
                      in zip(predicates, new_sample)
                      if pred_sample or pred_sample is None]
        
    return blocked_sample

def dedupeSamplePredicates(sample_size, predicates, items) :
    n_items = len(items)

    for subsample_size, predicate in subsample(sample_size, predicates) : 

        if not subsample_size :
            yield None
            continue

        items.rotate(random.randrange(n_items))

        try : # the reverse method was only added in python 2.7
            items.reverse()
        except AttributeError :
            items = deque(reversed(items))

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
                pair = sort_pair(block_dict.pop(block_key), index)
                sample.append(pair)
                subsample_size -= 1

                if subsample_size :
                    break
                else :
                    return sample

    else :
        return sample

def linkSamplePredicates(sample_size, predicates, items1, items2) :
    n_1 = len(items1)
    n_2 = len(items2)

    for subsample_size, predicate in subsample(sample_size, predicates) :
        
        if not subsample_size :
            yield None
            continue

        items1.rotate(random.randrange(n_1))
        items2.rotate(random.randrange(n_2))

        try :
            items1.reverse()
            items2.reverse()
        except AttributeError :
            items1 = deque(reversed(items1))
            items2 = deque(reversed(items2))

        yield linkSamplePredicate(subsample_size, predicate, items1, items2)


def linkSamplePredicate(subsample_size, predicate, items1, items2) :
    sample = []

    predicate_function = predicate.func
    field = predicate.field

    red = defaultdict(list)
    blue = defaultdict(list)

    for i, (index, record) in enumerate(interleave(items1, items2)):
        if i == 20000:
            if min(len(red), len(blue)) + len(sample) < 10 :
                return sample

        block_keys = predicate_function(record[field])
        for block_key in block_keys:
            if blue.get(block_key):
                pair = sort_pair(blue[block_key].pop(), index)
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break
                else :
                    return sample
            else:
                red[block_key].append(index)

        red, blue = blue, red

    for index, record in itertools.islice(items2, len(items1)) :
        block_keys = predicate_function( record[field] )
        for block_key in block_keys:
            if red.get(block_key):
                pair = sort_pair(red[block_key].pop(), index)
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break
                else :
                    return sample

    return sample

def evenSplits(total_size, num_splits) :
    avg = total_size/float(num_splits) 
    split = 0
    for _ in xrange(num_splits) :
        split += avg - int(split)
        yield int(split)

def subsample(total_size, predicates) :
    splits = evenSplits(total_size, len(predicates))
    for split, predicate in zip(splits, predicates) :
        yield split, predicate

def interleave(*iterables) :
    return itertools.chain.from_iterable(itertools.izip(*iterables))

def sort_pair(a, b) :
    if a > b :
        return (b, a)
    else :
        return (a, b)

def randomDeque(data) :
    try : # viewitems comes in at python 2.7
        data_q = deque(random.sample(data.viewitems(), len(data)))
    except AttributeError : 
        data_q = deque(random.sample(data.items(), len(data)))
    
    return data_q


dedupeBlockedSample = functools.partial(blockedSample, dedupeSamplePredicates) 
linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) 


