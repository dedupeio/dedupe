#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
dedupe provides the main user interface for the library the
Dedupe class
"""

import itertools
import logging
import pickle
import multiprocessing
import warnings
import os
from collections import OrderedDict

import numpy
import json
import rlr

import dedupe.core as core
import dedupe.serializer as serializer
import dedupe.blocking as blocking
import dedupe.clustering as clustering
import dedupe.datamodel as datamodel
import dedupe.labeler as labeler
import dedupe.predicates

from typing import (Mapping,
                    Optional,
                    Dict,
                    List,
                    Tuple,
                    Union,
                    Generator,
                    Iterable,
                    Sequence,
                    Set,
                    Any,
                    BinaryIO,
                    Callable,
                    TextIO)
from dedupe._typing import (Data,
                            Clusters,
                            RecordPairs,
                            RecordID,
                            Record,
                            Blocks,
                            RecordDict,
                            TrainingExample,
                            LookupResults,
                            SearchResults,
                            TrainingData)

logger = logging.getLogger(__name__)


class Matching(object):
    """
    Base Class for Record Matching Classes

    Public methods:

    - `__init__`
    - `thresholdBlocks`
    - `matchBlocks`
    """

    def __init__(self, num_cores: Optional[int], **kwargs) -> None:
        if num_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = num_cores

        self.loaded_indices = False
        self.blocker: Optional[blocking.Blocker] = None
        self.data_model: datamodel.DataModel
        self.classifier: Any
        self.predicates: Iterable[dedupe.predicates.Predicate]
        self._cluster: Callable

    def threshold_pairs(self,
                        pairs: RecordPairs,
                        recall_weight: float = 1.5) -> float:  # pragma: nocover
        """
        Returns the threshold that maximizes the expected `F score
        <https://en.wikipedia.org/wiki/F1_score>`_, a weighted average
        of precision and recall for a sample of record pairs.

        Args:
            pairs: Sequence of pairs of records to compare
            recall_weight: Sets the tradeoff between precision and
                           recall. I.e. if you care twice as much about
                           recall as you do precision, set recall_weight
                           to 2.

        """
        probability = core.scoreDuplicates(pairs,
                                           self.data_model,
                                           self.classifier,
                                           self.num_cores)['score']

        probability = probability.copy()
        probability.sort()
        probability = probability[::-1]

        expected_dupes = numpy.cumsum(probability)

        recall = expected_dupes / expected_dupes[-1]
        precision = expected_dupes / numpy.arange(1, len(expected_dupes) + 1)

        score = recall * precision / (recall + recall_weight ** 2 * precision)

        i = numpy.argmax(score)

        logger.info('Maximum expected recall and precision')
        logger.info('recall: %2.3f', recall[i])
        logger.info('precision: %2.3f', precision[i])
        logger.info('With threshold: %2.3f', probability[i])

        return probability[i]

    def score(self,
              pairs: RecordPairs,
              threshold: float = 0.0) -> numpy.ndarray:
        """
        Scores pairs of records. Returns pairs of tuples of records id and
        associated probabilites that the pair of records are match

        Args:
            pairs: Iterator of pairs of records

            threshold: Number between 0 and 1 (default is .5). We will
                       only consider as duplicates record pairs as
                       duplicates if their estimated duplicate
                       likelihood is greater than the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

        """

        matches = core.scoreDuplicates(pairs,
                                       self.data_model,
                                       self.classifier,
                                       self.num_cores,
                                       threshold=threshold)

        return matches


class DedupeMatching(Matching):
    """
    Class for Deduplication, extends Matching.

    Use DedupeMatching when you have a dataset that can contain
    multiple references to the same entity.

    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cluster = clustering.cluster

    def match(self,
              data: Data,
              threshold: float = 0.5,
              generator: bool = False) -> Clusters:  # pragma: no cover
        """
        Identifies records that all refer to the same entity, returns
        tuples containing a sequence of record ids and corresponding
        sequence of confidence score as a float between 0 and 1. The
        record_ids within each set should refer to the same entity and the
        confidence score is a measure of our confidence a particular entity
        belongs in the cluster.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to the :func:`~score`.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

            threshold: Number between 0 and 1 (default is .5). We will
                       consider records as potential duplicates if the
                       predicted probability of being a duplicate is
                       above the threshold.

                       Lowering the number will increase recall,
                       raising it will increase precision

            generator: Should :func:`match` return a list of clusters or a generator

        .. code:: python

           > clusters = matcher.match(data, threshold=0.5)
           > print(duplicates)
           [((1, 2, 3), (0.790, 0.860, 0.790)),
             ((4, 5), (0.720, 0.720)),
             ((10, 11), (0.899, 0.899))]
        """
        pairs = self.pairs(data)
        pair_scores = self.score(pairs)
        clusters = self.cluster(pair_scores, threshold)

        try:
            if generator:
                return clusters
            else:
                clusters = list(clusters)
                return clusters
        finally:
            try:
                mmap_file = pair_scores.filename
                del pair_scores
                os.remove(mmap_file)
            except AttributeError:
                pass

    def threshold(self,
                  data: Data,
                  recall_weight: float = 1.5) -> float:  # pragma: no cover
        """
        Returns the threshold that maximizes the expected `F score
        <https://en.wikipedia.org/wiki/F1_score>`_,
        a weighted average of precision and recall for a sample of
        data.

        Args:
            data: Dictionary of records, where the keys are record_ids
                  and the values are dictionaries with the keys being
                  field names

            recall_weight: Sets the tradeoff between precision and
                           recall. I.e. if you care twice as much about
                           recall as you do precision, set recall_weight
                           to 2.
        """

        pairs = self.pairs(data)
        return self.threshold_pairs(pairs, recall_weight)

    def pairs(self, data_d: Data) -> RecordPairs:
        """
        TK
        """

        assert self.blocker
        if not self.loaded_indices:
            self.blocker.indexAll(data_d)

        blocked_records: Dict[str, List[RecordID]] = {}
        for block_key, record_id in self.blocker(data_d.items()):
            blocked_records.setdefault(block_key, []).append(record_id)

        block_groups = itertools.groupby(self.blocker(data_d.items()),
                                         lambda x: x[1])

        product = itertools.product

        for a_record_id, block_keys in block_groups:

            A: List[Record] = [(a_record_id, data_d[a_record_id])]

            b_record_ids: set = set()
            for block_key, _ in block_keys:
                if block_key in blocked_records:
                    b_record_ids.update(blocked_records[block_key])

            B: List[Record] = [(b_record_id, data_d[b_record_id])
                               for b_record_id
                               in b_record_ids
                               if b_record_id > a_record_id]

            if B:
                yield from product(A, B)

        if not self.loaded_indices:
            self.blocker.resetIndices()

    def cluster(self,
                matches: numpy.ndarray,
                threshold: float,
                *args,
                **kwargs) -> Clusters:
        """
        CLUSTERING
        """

        logger.debug("matching done, begin clustering")

        yield from self._cluster(matches, threshold, *args, **kwargs)


class RecordLinkMatching(Matching):
    """
    Class for Record Linkage, extends Matching.

    Use RecordLinkMatching when you have two datasets that you want to merge
    where each dataset, individually, contains no duplicates.

    Public methods:

    - `__init__`
    - `match`
    - `threshold`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._cluster = clustering.greedyMatching

    def match(self,
              data_1: Data,
              data_2: Data,
              threshold: float = 0.5,
              generator: bool = False,
              **kwargs) -> SearchResults:  # pragma: no cover
        """
        Identifies pairs of records that refer to the same entity, returns
        tuples containing a set of record ids and a confidence score as a float
        between 0 and 1. The record_ids within each set should refer to the
        same entity and the confidence score is the estimated probability that
        the records refer to the same entity.

        This method should only used for small to moderately sized
        datasets for larger data, you need may need to generate your
        own pairs of records and feed them to the :func:`~score`.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names

            data_2: Dictionary of records from second dataset, same form
                    as data_1

            threshold: Number between 0 and 1 (default is .5). We
                       will consider records as potential
                       duplicates if the predicted probability of
                       being a duplicate is above the threshold.

                       Lowering the number will increase recall, raising it
                       will increase precision

            generator: Should :func:`match` return a list of clusters or a generator

        """
        pairs = self.pairs(data_1, data_2)
        pair_scores = self.score(pairs, threshold)
        clusters = self.cluster(pair_scores, **kwargs)

        try:
            if generator:
                return clusters
            else:
                clusters = list(clusters)
                return clusters
        finally:
            try:
                mmap_file = pair_scores.filename
                del pair_scores
                os.remove(mmap_file)
            except AttributeError:
                pass

    def threshold(self,
                  data_1: Data,
                  data_2: Data,
                  recall_weight: float = 1.5) -> float:  # pragma: no cover
        """
        Returns the threshold that maximizes the expected `F score
        <https://en.wikipedia.org/wiki/F1_score>`_,
        a weighted average of precision and recall for sample of data sets.

        :param data_1: Dictionary of records from first dataset, where the
                       keys are record_ids and the values are dictionaries
                       with the keys being field names
        :param data_2: Dictionary of records from second dataset, same form
                       as data_1

        :param recall_weight: Sets the tradeoff between precision and
                              recall. I.e. if you care twice as much about
                              recall as you do precision, set recall_weight
                              to 2.
        """
        pairs = self.pairs(data_1, data_2)
        return self.threshold_pairs(pairs, recall_weight)

    def pairs(self, data_1: Data, data_2: Data) -> RecordPairs:
        """
        TK
        """

        assert self.blocker

        if not self.loaded_indices:
            self.blocker.indexAll(data_2)

        blocked_records: Dict[str, List[RecordID]] = {}

        for block_key, record_id in self.blocker(data_2.items(), target=True):
            blocked_records.setdefault(block_key, []).append(record_id)

        block_groups = itertools.groupby(self.blocker(data_1.items()),
                                         lambda x: x[1])

        product = itertools.product

        for i, (a_record_id, block_keys) in enumerate(block_groups):
            if i % 100 == 0:
                logger.info("%s records" % i)

            A: List[Record] = [(a_record_id, data_1[a_record_id])]

            b_record_ids: set = set()
            for block_key, _ in block_keys:
                if block_key in blocked_records:
                    b_record_ids.update(blocked_records[block_key])

            B: List[Record] = [(rec_id, data_2[rec_id])
                               for rec_id
                               in b_record_ids]

            if B:
                yield from product(A, B)

    def cluster(self,
                matches: numpy.ndarray,
                **kwargs) -> SearchResults:
        """
        TK
        """

        logger.debug("matching done, begin clustering")

        yield from self._cluster(matches, **kwargs)


class GazetteerMatching(RecordLinkMatching):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._cluster = clustering.pair_gazette_matching
        self.blocked_records: Dict[str, Dict[RecordID, RecordDict]] = {}

    def index(self, data: Data) -> None:  # pragma: no cover
        """
        Add records to the index of records to match against. If a record in
        `canonical_data` has the same key as a previously indexed record, the
        old record will be replaced.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        assert self.blocker

        self.blocker.indexAll(data)

        for block_key, record_id in self.blocker(data.items(), target=True):
            if block_key not in self.blocked_records:
                self.blocked_records[block_key] = {}
            self.blocked_records[block_key][record_id] = data[record_id]

    def unindex(self, data: Data) -> None:  # pragma: no cover
        """
        Remove records from the index of records to match against.

        Args:
            data: a dictionary of records where the keys
                  are record_ids and the values are
                  dictionaries with the keys being
                  field_names
        """

        assert self.blocker

        for field in self.blocker.index_fields:
            self.blocker.unindex((record[field]
                                  for record
                                  in data.values()),
                                 field)

        for block_key, record_id in self.blocker(data.items()):
            try:
                del self.blocked_records[block_key][record_id]
            except KeyError:
                pass

    def blocks(self, data_1: Data) -> Blocks:

        assert self.blocker

        block_groups = itertools.groupby(self.blocker(data_1.items()),
                                         lambda x: x[1])

        product = itertools.product

        for i, (a_record_id, block_keys) in enumerate(block_groups):
            if i % 100 == 0:
                logger.info("%s records" % i)

            A: List[Record] = [(a_record_id, data_1[a_record_id])]

            B: Dict[RecordID, RecordDict] = {}
            for block_key, _ in block_keys:
                if block_key in self.blocked_records:
                    B.update(self.blocked_records[block_key])

            if B:
                yield product(A, B.items())

    def score_blocks(self,
                     blocks: Blocks,
                     threshold: float,
                     **kwargs) -> Generator[numpy.ndarray, None, None]:
        """
        TK
        """

        matches = core.scoreGazette(blocks,
                                    self.data_model,
                                    self.classifier,
                                    self.num_cores,
                                    threshold=threshold)

        return matches

    def cluster_blocks(self,
                       score_blocks: Iterable[numpy.ndarray],
                       n_matches: int = 1) -> SearchResults:
        """
        TK
        """

        yield from clustering.gazetteMatching(score_blocks, n_matches)

    def search(self,
               messy_data: Data,
               threshold: float = 0.5,
               n_matches: int = 1,
               generator: bool = False) -> LookupResults:  # pragma: no cover
        """
        Identifies pairs of records that could refer to the same entity,
        returns tuples containing tuples of possible matches, with a
        confidence score for each match. The record_ids within each
        tuple should refer to potential matches from a messy data
        record to canonical records. The confidence score is the
        estimated probability that the records refer to the same
        entity.

        Args:

            messy_data: a dictionary of records from a messy
                        dataset, where the keys are record_ids and
                        the values are dictionaries with the keys
                        being field names.

            threshold: a number between 0 and 1 (default is
                       0.5). We will consider records as
                       potential duplicates if the predicted
                       probability of being a duplicate is
                       above the threshold.

                       Lowering the number will increase
                       recall, raising it will increase
                       precision
            n_matches: the maximum number of possible matches from
                       canonical_data to return for each record in
                       messy_data. If set to `None` all possible
                       matches above the threshold will be
                       returned. Defaults to 1
            generator: when `True`, match will generate a sequence of
                       possible matches, instead of a list. Defaults
                       to `False` This makes `match` a lazy method.

        .. code:: python

            > matches = gazetteer.search(messy_data, threshold=0.5, n_matches=2)
            > print(matches)
            [(((1, 6), 0.72),
              ((1, 8), 0.6)),
             (((2, 7), 0.72),),
             (((3, 6), 0.72),
              ((3, 8), 0.65)),
             (((4, 6), 0.96),
              ((4, 5), 0.63))]

        """
        blocks = self.blocks(messy_data)
        pair_scores = self.score_blocks(blocks, threshold=threshold)
        search_results = self.cluster_blocks(pair_scores, n_matches)

        results = self._format_search_results(messy_data, search_results)

        if generator:
            return results
        else:
            return list(results)

    def write_settings(self,
                       file_obj: BinaryIO,
                       index: bool = False) -> None:  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        Args:
            file_obj: file object to write settings data into
        """
        super().write_settings(file_obj, index)

        if index:
            pickle.dump(self.blocked_records, file_obj)

    def _format_search_results(self,
                               search_d: Data,
                               results: SearchResults) -> LookupResults:

        seen: Set[RecordID] = set()

        for result in results:
            a = None
            prepared_result = []
            for (a, b), score in result:  # type: ignore
                prepared_result.append((b, score))
            yield a, tuple(prepared_result)

        for k in (search_d.keys() - seen):
            yield k, ()


class StaticMatching(Matching):
    """
    Class for initializing a dedupe object from a settings file,
    extends Matching.

    Public methods:
    - __init__
    """

    def __init__(self,
                 settings_file: BinaryIO,
                 num_cores: int = None,
                 **kwargs) -> None:  # pragma: no cover
        """
        :param settings_file: A file object containing settings
                              info produced from the
                              :func:`~dedupe.api.ActiveMatching.write_settings` method.
        :param num_cores: the number of cpus to use for parallel
                          processing, defaults to the number of cpus
                          available on the machine
        """
        super().__init__(num_cores, **kwargs)

        try:
            self.data_model = pickle.load(settings_file)
            self.classifier = pickle.load(settings_file)
            self.predicates = pickle.load(settings_file)
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")

        try:
            self._loadIndices(settings_file)
        except EOFError:
            pass
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")

        logger.info(self.predicates)

        self.blocker = blocking.Blocker(self.predicates)

    def _loadIndices(self, settings_file: BinaryIO) -> None:
        canopies = pickle.load(settings_file)
        indices = pickle.load(settings_file)
        doc_to_ids = pickle.load(settings_file)

        for full_predicate in self.predicates:
            for predicate in full_predicate:
                if hasattr(predicate, "index") and predicate.index is None:
                    predicate.index = predicate.initIndex()
                    max_id = max(doc_to_ids[predicate].values())
                    predicate.index._doc_to_id = core.Enumerator(max_id + 1,
                                                                 doc_to_ids[predicate])

                    if hasattr(predicate, "canopy"):
                        predicate.canopy = canopies[predicate]
                    else:
                        try:
                            predicate.index._index = indices[predicate]
                        except KeyError:
                            pass

        self.loaded_indices = True


class ActiveMatching(Matching):
    """
    Class for training dedupe extends Matching.

    Public methods:
    - __init__
    - train
    - write_settings
    - write_training
    - uncertainPairs
    - mark_pairs
    - cleanup_training
    """
    classifier = rlr.RegularizedLogisticRegression()

    def __init__(self,
                 variable_definition: Sequence[Mapping],
                 num_cores: int = None,
                 **kwargs) -> None:
        """
        :param variable_definition: A list of dictionaries describing
                                    the variables will be used for
                                    training a model. **add link**

        :param num_cores: the number of cpus to use for parallel
                          processing. If set to `None`, uses all cpus
                          available on the machine.
        """
        super().__init__(num_cores, **kwargs)

        self.data_model = datamodel.DataModel(variable_definition)

        self.training_pairs: TrainingData
        self.training_pairs = OrderedDict({u'distinct': [],
                                           u'match': []})
        self.active_learner: Optional[Union[labeler.DedupeDisagreementLearner,
                                            labeler.RecordLinkDisagreementLearner]]
        self.active_learner = None

    def cleanup_training(self) -> None:  # pragma: no cover
        '''
        Clean up data we used for training. Free up memory.
        '''
        del self.training_pairs
        del self.active_learner

    def _read_training(self, training_file: TextIO) -> None:
        '''
        Read training from previously built training data file object

        Args:
            training_file: file object containing the training data
        '''

        logger.info('reading training from file')
        training_pairs = json.load(training_file,
                                   cls=serializer.dedupe_decoder)

        try:
            self.mark_pairs(training_pairs)
        except AttributeError as e:
            if "Attempting to block with an index predicate without indexing records" in str(e):
                raise UserWarning('Training data has records not known '
                                  'to the active learner. Read training '
                                  'in before initializing the active '
                                  'learner with the sample method, or '
                                  'use the prepare_training method.')
            else:
                raise

    def train(self,
              recall: float = 0.95,
              index_predicates: bool = True) -> None:  # pragma: no cover
        """
        Learn final pairwise classifier and blocking rules. Requires that
        adequate training data has been already been provided.

        Args:
            recall: The proportion of true dupe pairs in our
                    training data that that the learned blocks
                    must cover. If we lower the recall, there will
                    be pairs of true dupes that we will never
                    directly compare.

                    recall should be a float between 0.0 and 1.0.

            index_predicates: Should dedupe consider predicates
                              that rely upon indexing the
                              data. Index predicates can be slower
                              and take substantial memory.

        """
        assert self.active_learner, "Please initialize with the sample method"

        examples, y = flatten_training(self.training_pairs)
        self.classifier.fit(self.data_model.distances(examples), y)

        self.predicates = self.active_learner.learn_predicates(
            recall, index_predicates)
        self.blocker = blocking.Blocker(self.predicates)
        self.blocker.resetIndices()

    def write_training(self, file_obj: TextIO) -> None:  # pragma: no cover
        """
        Write to a json file that contains labeled examples

        :param file_obj: file object to write training data to

        .. code:: python

            with open('training.json', 'w') as f:
                matcher.write_training(f)

        """

        json.dump(self.training_pairs,
                  file_obj,
                  default=serializer._to_json,
                  ensure_ascii=True)

    def write_settings(self,
                       file_obj: BinaryIO,
                       index: bool = False) -> None:  # pragma: no cover
        """
        Write a settings file containing the
        data model and predicates to a file object

        :param file_obj: file object to write settings data into

        .. code:: python

           with open('learned_settings', 'wb') as f:
               deduper.write_settings(f)

        """

        pickle.dump(self.data_model, file_obj)
        pickle.dump(self.classifier, file_obj)
        pickle.dump(self.predicates, file_obj)

        if index:
            self._writeIndices(file_obj)

    def _writeIndices(self, file_obj: BinaryIO) -> None:
        indices = {}
        doc_to_ids = {}
        canopies = {}
        for full_predicate in self.predicates:
            for predicate in full_predicate:
                if hasattr(predicate, 'index') and predicate.index:
                    doc_to_ids[predicate] = dict(predicate.index._doc_to_id)
                    if hasattr(predicate, "canopy"):
                        canopies[predicate] = predicate.canopy
                    else:
                        try:
                            indices[predicate] = predicate.index._index
                        except AttributeError:
                            pass

        pickle.dump(canopies, file_obj)
        pickle.dump(indices, file_obj)
        pickle.dump(doc_to_ids, file_obj)

    def uncertain_pairs(self) -> TrainingExample:
        '''
        Returns a list of pairs of records from the sample of record pairs
        tuples that Dedupe is most curious to have labeled.

        This method is mainly useful for building a user interface for training
        a matching model.

       .. code:: python

          > pair = matcher.uncertainPairs()
          > print(pair)
          [({'name' : 'Georgie Porgie'}, {'name' : 'Georgette Porgette'})]

        '''
        assert self.active_learner, "Please initialize with the sample method"
        return self.active_learner.pop()

    def mark_pairs(self, labeled_pairs: TrainingData) -> None:
        '''
        Add users labeled pairs of records to training data and update the
        matching model

        This method is useful for building a user interface for training a
        matching model or for adding training data from an existing source.

        Args:
            labeled_pairs: A dictionary with two keys, `match` and `distinct`
                           the values are lists that can contain pairs of
                           records

        .. code:: python

            labeled_examples = {'match'    : [],
                                'distinct' : [({'name' : 'Georgie Porgie'},
                                               {'name' : 'Georgette Porgette'})]
                                }
            matcher.mark_pairs(labeled_examples)

        '''
        self._checkTrainingPairs(labeled_pairs)

        for label, examples in labeled_pairs.items():
            self.training_pairs[label].extend(examples)

        if self.active_learner:
            examples, y = flatten_training(labeled_pairs)
            self.active_learner.mark(examples, y)

    def _checkTrainingPairs(self, labeled_pairs: TrainingData) -> None:
        try:
            labeled_pairs.items()
            labeled_pairs[u'match']
            labeled_pairs[u'distinct']
        except (AttributeError, KeyError):
            raise ValueError('labeled_pairs must be a dictionary with keys '
                             '"distinct" and "match"')

        if labeled_pairs[u'match']:
            pair = labeled_pairs[u'match'][0]
            self._checkRecordPair(pair)

        if labeled_pairs[u'distinct']:
            pair = labeled_pairs[u'distinct'][0]
            self._checkRecordPair(pair)

        if not labeled_pairs[u'distinct'] and not labeled_pairs[u'match']:
            warnings.warn("Didn't return any labeled record pairs")

    def _checkRecordPair(self, record_pair: TrainingExample) -> None:
        try:
            a, b = record_pair
        except ValueError:
            raise ValueError("The elements of data_sample must be pairs "
                             "of record_pairs")
        try:
            record_pair[0].keys() and record_pair[1].keys()
        except AttributeError:
            raise ValueError("A pair of record_pairs must be made up of two "
                             "dictionaries ")

        self.data_model.check(record_pair[0])
        self.data_model.check(record_pair[1])


class StaticDedupe(StaticMatching, DedupeMatching):
    """
    Class for deduplication using saved settings. If you have already
    trained a :class:`Dedupe` object and saved the settings, you can
    load the saved settings with StaticDedupe.

    """


class Dedupe(ActiveMatching, DedupeMatching):
    """
    Class for active learning deduplication. Use deduplication when you have
    data that can contain multiple records that can all refer to the same
    entity.
    """

    canopies = True
    ActiveLearner = labeler.DedupeDisagreementLearner

    def prepare_training(self,
                         data: Data,
                         training_file: TextIO = None,
                         sample_size: int = 1500,
                         blocked_proportion: float = 0.9,
                         original_length: int = None) -> None:
        '''
        Initialize the active learner with your data and, optionally,
        existing training data.

        Sets up the learner.

        Args:
            data: Dictionary of records, where the keys are
                  record_ids and the values are dictionaries
                  with the keys being field names
            training_file: file object containing training data
            sample_size: Size of the sample to draw
            blocked_proportion: Proportion of the sample that will be blocked
            original_length: If `data` is a subsample of all your data,
                             `original_length` should be the size of
                             your complete data. By default,
                             `original_length` defaults to the length of
                             `data`.

        .. code:: python

           matcher.prepare_training(data_d, 150000, .5)

           # or
           with open('training_file.json') as f:
               matcher.prepare_training(data_d, training_file=f)


        '''

        if training_file:
            self._read_training(training_file)
        self._sample(data, sample_size, blocked_proportion, original_length)

    def _sample(self,
                data: Data,
                sample_size: int = 15000,
                blocked_proportion: float = 0.5,
                original_length: int = None) -> None:
        '''Draw a sample of record pairs from the dataset
        (a mix of random pairs & pairs of similar records)
        and initialize active learning with this sample


        :param data: Dictionary of records, where the keys are
                     record_ids and the values are dictionaries
                     with the keys being field names

        :param sample_size: Size of the sample to draw

        :param blocked_proportion: Proportion of the sample that will be blocked

        :param original_length: Length of original data, should be set
                                if `data` is a sample of full data

        '''
        self._checkData(data)

        if not original_length:
            original_length = len(data)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionary
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length,
                                                 index_include=examples)

        self.active_learner.mark(examples, y)

    def _checkData(self, data: Data) -> None:
        if len(data) == 0:
            raise ValueError(
                'Dictionary of records is empty.')

        self.data_model.check(next(iter(data.values())))


class Link(ActiveMatching):
    """
    Mixin Class for Active Learning Record Linkage

    Public Methods
    - sample
    - prepare_training
    """

    canopies = False
    ActiveLearner = labeler.RecordLinkDisagreementLearner

    def prepare_training(self,
                         data_1: Data,
                         data_2: Data,
                         training_file: Optional[TextIO] = None,
                         sample_size: int = 15000,
                         blocked_proportion: float = 0.5,
                         original_length_1: Optional[int] = None,
                         original_length_2: Optional[int] = None) -> None:
        '''
        Initialize the active learner with your data and, optionally,
        existing training data.

        Args:
            data_1: Dictionary of records from first dataset, where the
                    keys are record_ids and the values are dictionaries
                    with the keys being field names
            data_2: Dictionary of records from second dataset, same
                    form as data_1
            training_file: file object containing training data

            sample_size: The size of the sample to draw. Defaults to 150,000

            blocked_proportion: The proportion of record pairs to
                                be sampled from similar records,
                                as opposed to randomly selected
                                pairs. Defaults to 0.5.
            original_length_1: If `data_1` is a subsample of your first dataset,
                               `original_length_1` should be the size of
                               the complete first dataset. By default,
                               `original_length_1` defaults to the length of
                               `data_1`
            original_length_2: If `data_2` is a subsample of your first dataset,
                               `original_length_2` should be the size of
                               the complete first dataset. By default,
                               `original_length_2` defaults to the length of
                               `data_2`

        .. code:: python

           matcher.prepare_training(data_1, data_2, 150000)

           with open('training_file.json') as f:
               matcher.prepare_training(data_1, data_2, training_file=f)

        '''

        if training_file:
            self._read_training(training_file)
        self._sample(data_1,
                     data_2,
                     sample_size,
                     blocked_proportion,
                     original_length_1,
                     original_length_2)

    def _sample(self,
                data_1: Data,
                data_2: Data,
                sample_size: int = 15000,
                blocked_proportion: float = 0.5,
                original_length_1: int = None,
                original_length_2: int = None) -> None:
        '''
        Draws a random sample of combinations of records from
        the first and second datasets, and initializes active
        learning with this sample

        :param data_1: Dictionary of records from first dataset, where the
                       keys are record_ids and the values are dictionaries
                       with the keys being field names
        :param data_2: Dictionary of records from second dataset, same
                       form as data_1
        :param sample_size: Size of the sample to draw
        '''
        self._checkData(data_1, data_2)

        # We need the active learner to know about all our
        # existing training data, so add them to data dictionaries
        examples, y = flatten_training(self.training_pairs)

        self.active_learner = self.ActiveLearner(self.data_model,
                                                 data_1,
                                                 data_2,
                                                 blocked_proportion,
                                                 sample_size,
                                                 original_length_1,
                                                 original_length_2,
                                                 index_include=examples)

        self.active_learner.mark(examples, y)

    def _checkData(self, data_1: Data, data_2: Data) -> None:
        if len(data_1) == 0:
            raise ValueError(
                'Dictionary of records from first dataset is empty.')
        elif len(data_2) == 0:
            raise ValueError(
                'Dictionary of records from second dataset is empty.')

        self.data_model.check(next(iter(data_1.values())))
        self.data_model.check(next(iter(data_2.values())))


class RecordLink(Link, RecordLinkMatching):
    """
    Class for active learning record linkage.

    Use RecordLinkMatching when you have two datasets that you want to
    merge. Each dataset, individually, should contain no duplicates. A
    record from the first dataset can match one and only one record from the
    second dataset and vice versa. A record from the first dataset need not
    match any record from the second dataset and vice versa.
    """


class StaticRecordLink(StaticMatching, RecordLinkMatching):
    """
    Class for record linkage using saved settings. If you have already
    trained a RecordLink instance, you can load the saved settings with
    StaticRecordLink.
    """


class Gazetteer(Link, GazetteerMatching):
    """
    Class for active learning gazetteer matching.

    Gazetteer matching is for matching a messy data set against a
    'canonical dataset', i.e. one that does not have any
    duplicates. This class is useful for such tasks as matching messy
    addresses against a clean list
    """

    def __init__(self,
                 variable_definition: Sequence[Mapping],
                 num_cores: int = None,
                 **kwargs) -> None:  # pragma: no cover
        super().__init__(variable_definition, num_cores=num_cores, **kwargs)
        self.blocked_records = OrderedDict({})


class StaticGazetteer(StaticMatching, GazetteerMatching):
    """
    Class for gazetter matching using saved settings.

    If you have already trained a :class:`Gazetteer` instance, you can
    load the saved settings with StaticGazetteer.
    """
    def __init__(self,
                 settings_file: BinaryIO,
                 num_cores: int = None,
                 **kwargs) -> None:  # pragma: no cover
        super().__init__(settings_file, num_cores=num_cores, **kwargs)

        try:
            self.blocked_records = pickle.load(settings_file)
        except EOFError:
            self.blocked_records = OrderedDict({})
        except (KeyError, AttributeError):
            raise SettingsFileLoadingException(
                "This settings file is not compatible with "
                "the current version of dedupe. This can happen "
                "if you have recently upgraded dedupe.")
        except:  # noqa: E722
            raise SettingsFileLoadingException(
                "Something has gone wrong with loading the settings file. "
                "Try deleting the file")


class EmptyTrainingException(Exception):
    pass


class SettingsFileLoadingException(Exception):
    pass


def flatten_training(training_pairs: TrainingData) -> Tuple[List[TrainingExample], numpy.ndarray]:
    examples = []
    y = []
    for label, pairs in training_pairs.items():
        for pair in pairs:
            if label == 'match':
                y.append(1)
                examples.append(pair)
            elif label == 'distinct':
                y.append(0)
                examples.append(pair)

    return examples, numpy.array(y)
