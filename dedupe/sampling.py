import numpy
import itertools

def blockedSample(indexed_data, predicates, sample_size) :

    indexed_items = indexed_data.items()
    indexed_items = numpy.array(indexed_items, dtype=object)
    numpy.random.shuffle(indexed_data)

    blocked_sample = []
    remaining_sample = sample_size

    while remaining_sample :
        new_sample = list(samplePredicates(indexed_items, 
                                           predicates,
                                           remaining_sample))
        
        blocked_sample.extend(itertools.chain.from_iterable(new_sample))
        
        predicates = list(itertools.compress(predicates, new_sample))

        remaining_sample = sample_size - len(blocked_sample)


    data_sample = tuple([(indexed_data[k1], indexed_data[k2]) 
                         for k1, k2 in blocked_sample])
        
    return data_sample

def samplePredicates(indexed_items, predicates, sample_size) :

    subsample_counts = evenSplits(sample_size, len(predicates))

    requested_samples = [(count, predicate) 
                         for count, predicate
                         in zip(subsample_counts, predicates)
                         if count]
    n_items = len(indexed_items)

    for subsample_size, predicate in requested_samples :

        indexed_items = numpy.roll(indexed_items, 
                                   numpy.random.randint(n_items), 
                                   0)

        yield samplePredicate(subsample_size,
                              predicate,
                              indexed_items)
        
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
