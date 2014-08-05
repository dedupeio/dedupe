#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import collections
import itertools
import logging
import backport
from zope.index.text.textindex import TextIndex
from zope.index.text.cosineindex import CosineIndex
from zope.index.text.lexicon import Lexicon
from zope.index.text.lexicon import Splitter
from zope.index.text.stopdict import get_stopdict
import time
import dedupe.tfidf as tfidf

logger = logging.getLogger(__name__)

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = None) :

        if stop_words is None :
            stop_words = defaultdict(lambda : set(get_stopdict()))

        self.predicates = predicates

        self.stop_words = stop_words

        self.tfidf_fields = defaultdict(set)

        for full_predicate in predicates :
            for predicate in full_predicate :
                if hasattr(predicate, 'canopy') :
                    self.tfidf_fields[predicate.field].add(predicate)

    #@profile
    def __call__(self, records):

        start_time = time.time()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        for i, record in enumerate(records) :
            record_id, instance = record
    
            for pred_id, predicate in predicates :
                block_keys = predicate(record_id, instance)
                for block_key in block_keys :
                    yield block_key + pred_id, record_id
            
            if i % 10000 == 0 :
                logger.info('%(iteration)d, %(elapsed)f2 seconds',
                             {'iteration' :i,
                              'elapsed' :time.time() - start_time})

        self._resetCanopies()


    def _resetCanopies(self) :
        # clear canopies to reduce memory usage
        for predicate_set in self.tfidf_fields.values() :
            for predicate in predicate_set :
                predicate.canopy = {}



class DedupeBlocker(Blocker) :
    

    def tfIdfBlock(self, data, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        splitter = Splitter()

        stop_word_remover = CustomStopWordRemover(self.stop_words[field])


        indices = {}
        for predicate in self.tfidf_fields[field] :
            indices[predicate] = TextIndex(Lexicon(splitter, stop_word_remover))
            indices[predicate].index = CosineIndex(indices[predicate].lexicon)
            pipeline = indices[predicate].lexicon._pipeline
            stringify = predicate.stringify

        index_to_id = {}
        base_tokens = {}

        for i, (record_id, doc) in enumerate(data, 1) :
            doc = stringify(doc)
            index_to_id[i] = record_id
            last = [doc]
            for each in pipeline :
                last = each.process(last)
            base_tokens[i] = ' OR '.join(last)
            for index in indices.values() :
                index.index_doc(i, doc)

        logger.info(time.asctime())                

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            canopy = tfidf.makeCanopy(indices[predicate],
                                      base_tokens, 
                                      predicate.threshold)
            predicate.canopy = dict((index_to_id[k], index_to_id[v])
                                    for k, v
                                    in canopy.iteritems())
        
        logger.info(time.asctime())                
               
class RecordLinkBlocker(Blocker) :
    def tfIdfBlock(self, data_1, data_2, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        splitter = Splitter()

        stop_word_remover = CustomStopWordRemover(self.stop_words[field])

        indices = {}
        for predicate in self.tfidf_fields[field] :
            indices[predicate] = TextIndex(Lexicon(splitter, stop_word_remover))
            indices[predicate].index = CosineIndex(indices[predicate].lexicon)
            pipeline = indices[predicate].lexicon._pipeline
            stringify = predicate.stringify

        index_to_id = {}
        base_tokens = {}

        i = 1

        for record_id, doc in data_1 :
            doc = stringify(doc)
            index_to_id[i] = record_id
            last = [doc]
            for each in pipeline :
                last = each.process(last)
            base_tokens[i] = ' OR '.join(last)
            i += 1

        for record_id, doc in data_2  :
            doc = stringify(doc)
            index_to_id[i] = record_id
            for index in indices.values() :
                index.index_doc(i, doc)
            i += 1

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            canopy = tfidf.makeCanopy(indices[predicate],
                                      base_tokens, 
                                      predicate.threshold)
            predicate.canopy = dict((index_to_id[k], index_to_id[v])
                                    for k, v
                                    in canopy.iteritems())


def blockTraining(training_pairs,
                  predicate_set,
                  eta=.1,
                  epsilon=.1,
                  matching = "Dedupe"):
    '''
    Takes in a set of training pairs and predicates and tries to find
    a good set of blocking rules.
    '''

    # Setup

    training_dupes = set(training_pairs['match'])
    training_distinct = set(training_pairs['distinct'])

    if matching == "RecordLink" :
        coverage = RecordLinkCoverage(predicate_set,
                                      training_dupes | training_distinct)

    else :
        coverage = DedupeCoverage(predicate_set,
                                  training_dupes | training_distinct)

    # Compound Predicates
    compound_predicates = itertools.combinations(coverage.overlap, 2)
    intersection = set.intersection

    for compound_predicate in compound_predicates :
        compound_predicate = CompoundPredicate(compound_predicate)
        predicate_1, predicate_2 = compound_predicate
        
        coverage.overlap[compound_predicate] =\
            intersection(coverage.overlap[predicate_1],
                         coverage.overlap[predicate_2])

        i = 0
        for blocks in itertools.product(coverage.blocks[predicate_1].values(),
                                        coverage.blocks[predicate_2].values()) :
            coverage.blocks[compound_predicate][i] =\
                intersection(*blocks)
            i += 1

    predicate_set = coverage.overlap.keys()
    
    # Only consider predicates that cover at least one duplicate pair
    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               training_dupes)
    predicate_set = dupe_coverage.keys()

    # Within blocks, we will compare every combination of
    # records. Therefore, we want to avoid predicates that make large
    # blocks.
    for pred in predicate_set[:] :
        blocks = coverage.blocks[pred].itervalues()
        if any(len(block) >= 100 for block in blocks if block) :
            predicate_set.remove(pred)

    # As an efficency, we can throw away the predicates that cover too
    # many distinct pairs
    coverage_threshold = eta * len(training_distinct)
    logger.info("coverage threshold: %s", coverage_threshold)

    distinct_coverage = coverage.predicateCoverage(predicate_set,
                                                   training_distinct)

    for pred, pairs in distinct_coverage.items() :
        if len(pairs) > coverage_threshold :
            predicate_set.remove(pred)

    distinct_coverage = coverage.predicateCoverage(predicate_set, 
                                                   training_distinct)

    final_predicate_set = findOptimumBlocking(training_dupes,
                                              predicate_set,
                                              distinct_coverage,
                                              epsilon,
                                              coverage)

    logger.info('Final predicate set:')
    for predicate in final_predicate_set :
        logger.info(predicate)

    if final_predicate_set:
        return final_predicate_set, coverage.stop_words
    else:
        raise ValueError('No predicate found! We could not learn a single good predicate. Maybe give Dedupe more training data')


def findOptimumBlocking(uncovered_dupes,
                        predicate_set,
                        distinct_coverage,
                        epsilon,
                        coverage):

    # Greedily find the predicates that, at each step, covers the
    # most duplicates and covers the least distinct pairs, due to
    # Chvatal, 1979
    #
    # We would like to optimize the ratio of the probability of of a
    # predicate covering a duplicate pair versus the probability of
    # covering a distinct pair. If we have a uniform prior belief
    # about those probabilities, we can estimate these probabilities as
    #
    # (predicate_covered_dupe_pairs + 1) / (all_dupe_pairs + 2)
    #
    # (predicate_covered_distinct_pairs + 1) / (all_distinct_pairs + 2)
    #
    # When we are trying to find the best predicate among a set of
    # predicates, the denominators factor out and our coverage
    # estimator becomes
    #
    # (predicate_covered_dupe_pairs + 1)/ (predicate_covered_distinct_pairs + 1)

    dupe_coverage = coverage.predicateCoverage(predicate_set,
                                               uncovered_dupes)
    
    uncovered_dupes = set(uncovered_dupes)

    final_predicate_set = []
    while len(uncovered_dupes) > epsilon:

        best_cover = 0
        best_predicate = None
        for predicate in dupe_coverage :
            dupes = len(dupe_coverage[predicate])
            distinct = len(distinct_coverage[predicate])
            cover = (dupes + 1.0)/(distinct + 1.0)
            if cover > best_cover:
                best_cover = cover
                best_predicate = predicate
                best_distinct = distinct
                best_dupes = dupes


        if not best_predicate:
            logger.warning('Ran out of predicates')
            break

        final_predicate_set.append(best_predicate)
        predicate_set.remove(best_predicate)
        
        uncovered_dupes = uncovered_dupes - dupe_coverage[best_predicate]
        dupe_coverage = coverage.predicateCoverage(predicate_set,
                                                   uncovered_dupes)


        logger.debug(best_predicate)
        logger.debug('cover: %(cover)f, found_dupes: %(found_dupes)d, '
                      'found_distinct: %(found_distinct)d, '
                      'uncovered dupes: %(uncovered)d',
                      {'cover' : best_cover,
                       'found_dupes' : best_dupes,
                       'found_distinct' : best_distinct,
                       'uncovered' : len(uncovered_dupes)
                       })

    return final_predicate_set

class Coverage(object) :

    def coveredBy(self, predicates, record_ids, pairs) :
        self.overlap = defaultdict(set)
        self.blocks = defaultdict(lambda : defaultdict(set))

        for pair in pairs :
            record_1, record_2 = pair
            record_1_id = record_ids[record_1]
            record_2_id = record_ids[record_2]
            for predicate in predicates :
                field_predicate_1 = predicate(record_1_id, record_1)
                if field_predicate_1:
                    field_predicate_2 = predicate(record_2_id, record_2)
                    if field_predicate_2 :
                        field_preds = (set(field_predicate_2) 
                                       & set(field_predicate_1))
                        if field_preds :
                            self.overlap[predicate].add(pair)
                            for field_pred in field_preds :
                                self.blocks[predicate][field_pred].add(pair)

        for predicate, coverage in self.blocks.items() :
            for field_pred, block in coverage.items() :
                if len(block) < 100 :
                    del self.blocks[predicate][field_pred]


    def predicateCoverage(self,
                          predicate_set,
                          pairs) :

        coverage = defaultdict(set)
        pairs = set(pairs)

        for predicate in predicate_set :
            covered_pairs = pairs.intersection(self.overlap[predicate])
            if covered_pairs :
                coverage[predicate] = covered_pairs

        return coverage

class DedupeCoverage(Coverage) :
    def __init__(self, predicate_set, pairs) :

        records = set(itertools.chain(*pairs))
        
        id_records = dict(itertools.izip(itertools.count(), records))
        record_ids = dict(itertools.izip(records, itertools.count()))


        blocker = DedupeBlocker(predicate_set)

        for field in blocker.tfidf_fields :
            data = [(record_id, record[field])
                    for record_id, record
                    in id_records.items()]
            stop_words = stopWords(data)
            blocker.stop_words[field].update(stop_words)
            blocker.tfIdfBlock(data, field)

        self.stop_words = blocker.stop_words
        self.coveredBy(blocker.predicates, record_ids, pairs)
        blocker._resetCanopies()

class RecordLinkCoverage(Coverage) :
    def __init__(self, predicate_set, pairs) :
        
        data_1 = set([])
        data_2 = set([])

        for record_1, record_2 in pairs :
            data_1.add(record_1)
            data_2.add(record_2)

        i = 0
        id_records_1 = {}
        id_records_2 = {}

        for record in data_1 :
            id_records_1[i] = record
            i += 1

        for record in data_2 :
            id_records_2[i] = record
            i += 1

        id_records = id_records_1.copy()
        id_records.update(id_records_2)
        
        record_ids = dict((record, record_id) 
                          for record_id, record
                          in id_records.items())

        blocker = RecordLinkBlocker(predicate_set)

        for field in blocker.tfidf_fields :
            fields_1 = ((record_id, record[field])
                        for record_id, record
                        in id_records_1.items())
            fields_2 = [(record_id, record[field])
                        for record_id, record
                        in id_records_2.items()]

            stop_words = stopWords(fields_2)
            blocker.stop_words[field].update(stop_words)
 
            blocker.tfIdfBlock(fields_1, fields_2, field)

        self.stop_words = blocker.stop_words
        self.coveredBy(blocker.predicates, record_ids, pairs)
        blocker._resetCanopies()
        

def stopWords(data) :
    index = TextIndex(Lexicon(Splitter()))

    for i, (_, doc) in enumerate(data, 1) :
        index.index_doc(i, doc)

    doc_freq = [(len(index.index._wordinfo[wid]), word) 
                for word, wid in index.lexicon.items()]

    doc_freq.sort(reverse=True)

    N = float(index.index.documentCount())
    threshold = int(max(1000, N * 0.05))

    stop_words = set([])

    for frequency, word in doc_freq :
        if frequency > threshold :
            stop_words.add(word)
        else :
            break

    return stop_words

class Predicate(object) :
    def __iter__(self) :
        yield self
        
    def __repr__(self) :
        return "%s: %s" % (self.type, self.__name__)

    def __hash__(self) :
        return hash(repr(self))

    def __eq__(self, other) :
        return repr(self) == repr(other)

class SimplePredicate(Predicate) :
    type = "SimplePredicate"

    def __init__(self, func, field) :
        self.func = func
        self.__name__ = "(%s, %s)" % (func.__name__, field)
        self.field = field

    def __call__(self, record_id, record) :
        column = record[self.field]
        return self.func(column)


class TfidfPredicate(Predicate):
    type = "TfidfPredicate"

    def __init__(self, threshold, field):
        self.__name__ = '(%s, %s)' % (threshold, field)
        self.field = field
        self.canopy = {}
        self.threshold = threshold

    def __call__(self, record_id, record) :
        center = self.canopy.get(record_id)
        if center is not None :
            return (unicode(center),)
        else :
            return ()


    def stringify(self, doc) :
        return doc


    def __getstate__(self):
        result = self.__dict__.copy()
        result['canopy'] = {}
        return result

class TfidfSetPredicate(TfidfPredicate) :
    type = "TfidfPredicate"

    def stringify(self, doc) :
        return ' '.join('_'.join(str(each).split()) for each in doc)


class CompoundPredicate(Predicate) :
    type = "CompoundPredicate"

    def __init__(self, predicates) :
        self.predicates = predicates
        self.__name__ = '(%s)' % ', '.join([str(pred)
                                            for pred in 
                                            predicates])

    def __iter__(self) :
        for pred in self.predicates :
            yield pred

    def __call__(self, record_id, record) :
        predicate_keys = (predicate(record_id, record)
                          for predicate in self.predicates)
        return (':'.join(block_key)
                for block_key
                in itertools.product(*predicate_keys))


class CustomStopWordRemover(object):
    def __init__(self, stop_words) :
        self.stop_words = stop_words

    def process(self, lst):
        return [w for w in lst if not w in self.stop_words]

