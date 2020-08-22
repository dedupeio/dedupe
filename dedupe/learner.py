# -*- coding: future_fstrings -*-

import random
from abc import ABC, abstractmethod
import logging
import numpy
import rlr
import functools
import dedupe.sampling as sampling
import dedupe.core as core
import dedupe.blocking as blocking
import dedupe.predicates as predicates
import itertools
import collections
from collections.abc import Mapping
from dedupe._typing import TrainingExample
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ActiveLearner(ABC):

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def pop(self) -> TrainingExample:
        pass

    @abstractmethod
    def mark(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class RLRLearner(ActiveLearner, rlr.RegularizedLogisticRegression):
    def __init__(self, distances, *args, **kwargs):
        """
        candidates: (list)(tuple(dict, dict)) A list of record pair tuples, e.g.
            ::
                [
                    ({record_1}, {record_2}),
                    ({record_2}, {record_5})
                ]
        """
        logger.info("Initializing RLRLearner class, calling super class ActiveLearner")
        super().__init__(alpha=1)

        self.distances = distances

        if 'candidates' not in kwargs:
            self.candidates = super().sample(*args)
        else:
            self.candidates = kwargs.pop('candidates')

        self.distance_matrix = self.transform(self.candidates)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])
        logger.info("Initializing fit transform with random pair")
        self.fit_transform([exact_match, random_pair],
                           [1, 0])

    def transform(self, pairs):
        return self.distances.compute_distance_matrix(pairs)

    def fit(self, X, y):
        """
        Args:
            X: (list)[list] a list of distance vectors, where the size of
                the distance vector is the same as the number of fields;
                distance between two records in R^n
            y: (list)[float] vector of either 1 or 0
        """
        self.y = numpy.array(y)
        self.X = X
        logger.debug("Fit model")
        super().fit(self.X, self.y, cv=False)

    def fit_transform(self, pairs, y):
        self.fit(self.transform(pairs), y)

    def pop(self) -> TrainingExample:
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        target_uncertainty = self._bias()

        probabilities = self.candidate_scores()

        distance_to_target = numpy.abs(target_uncertainty - probabilities)
        uncertain_index = distance_to_target.argmin()

        self.distances = numpy.delete(self.distances, uncertain_index, axis=0)

        uncertain_pair = self.candidates.pop(uncertain_index)

        return [uncertain_pair]  # AH upgrade

    def _remove(self, index):
        self.distances = numpy.delete(self.distances, index, axis=0)

    def mark(self, pairs, y):

        self.y = numpy.concatenate([self.y, y])
        self.X = numpy.vstack([self.X, self.transform(pairs)])

        self.fit(self.X, self.y)

    def _bias(self):
        positive = numpy.sum(self.y == 1)
        n_examples = len(self.y)

        bias = 1 - (positive / n_examples if positive else 0)

        # When we have just a few examples we are okay with getting
        # examples where the model strongly believes the example is
        # going to be positive or negative. As we get more examples,
        # prefer to ask for labels of examples the model is more
        # uncertain of.
        uncertainty_weight = min(positive, n_examples - positive)
        bias_weight = 10

        weighted_bias = 0.5 * uncertainty_weight + bias * bias_weight
        weighted_bias /= uncertainty_weight + bias_weight

        return weighted_bias

    def candidate_scores(self):
        return self.predict_proba(self.distances)

    def __len__(self):
        return len(self.candidates)


class DisagreementLearner(ActiveLearner):

    def _common_init(self):
        self.classifier = RLRLearner(self.distances,
                                     candidates=self.candidates)
        self.learners = (self.classifier, self.block_learner)
        self.y = numpy.array([])
        self.pairs = []

    def pop(self):
        if not len(self.candidates):
            raise IndexError("No more unlabeled examples to label")

        probs = []
        for learner in self.learners:
            probabilities = learner.candidate_scores()
            probs.append(probabilities)

        probs = numpy.concatenate(probs, axis=1)

        # where do the classifers disagree?
        disagreement = numpy.std(probs > 0.5, axis=1).astype(bool)

        if disagreement.any():
            conflicts = disagreement.nonzero()[0]
            target = numpy.random.uniform(size=1)
            uncertain_index = conflicts[numpy.argmax(probs[conflicts][:, 0] - target)]
        else:
            uncertain_index = numpy.std(probs, axis=1).argmax()

        logger.debug("Classifier: %.2f, Covered: %s",
                     probs[uncertain_index][0],
                     bool(probs[uncertain_index][1]))

        uncertain_pair = self.candidates.pop(uncertain_index)

        for learner in self.learners:
            learner._remove(uncertain_index)

        return [uncertain_pair]

    def mark(self, pairs, y):
        """
        Args:
            pairs: (list)[list] ordered list of all the record pairs (distinct and match)

                [
                    [record_1, record_2],
                    [record_1, record_3]
                ]
            y: (list)[int] list of either 1 or 0, corresponding to examples list
                1 = match
                0 = distinct
        """

        logger.debug("Fitting classifier with active label training data")
        self.y = numpy.concatenate([self.y, y])
        self.pairs.extend(pairs)

        for learner in self.learners:
            learner.fit_transform(self.pairs, self.y)

    def __len__(self):
        return len(self.candidates)

    def transform(self):
        pass

    def learn_predicates(self, recall, index_predicates):
        """
        Args:
            recall: (float)
            index_predicates: (boolean)
        """
        dupes = [pair for label, pair in zip(self.y, self.pairs) if label]

        if not index_predicates:
            logger.info(f"Copying predicates from blocking.Fingerprinter")
            old_preds = self.block_learner.fingerprinter.predicates.copy()

            no_index_predicates = [pred for pred in old_preds
                                   if not hasattr(pred, 'index')]
            self.block_learner.fingerprinter.predicates = no_index_predicates
            learned_preds = self.block_learner.learn(dupes, recall=recall)

            self.block_learner.fingerprinter.predicates = old_preds

        else:
            learned_preds = self.block_learner.learn(dupes, recall=recall)

        return learned_preds


class DedupeDisagreementLearner(DisagreementLearner):

    def __init__(self,
                 distances,
                 data,
                 blocked_proportion,
                 sample_size,
                 original_length,
                 index_include):

        self.distances = distances
        self.sampler = DedupeSampler(distances)
        data = core.index(data)

        self.candidates = self.sampler.sample(data, blocked_proportion, sample_size)

        random_pair = random.choice(self.candidates)
        exact_match = (random_pair[0], random_pair[0])

        index_include = index_include.copy()
        index_include.append(exact_match)

        self.block_learner = BlockLearner(distances,
                                          self.candidates,
                                          data,
                                          original_length,
                                          index_include)

        self._common_init()
        logger.debug("Initializing with 5 random values")
        self.mark([exact_match] * 4 + [random_pair],
                  [1] * 4 + [0])


class BlockLearner(object):

    def __init__(self, distances, candidates, data, original_length, index_include, *args):
        """
        simple_cover: (dict) subset of the predicates list
            {
                key: (dedupe.predicates class)
                value: (dedupe.training.Counter)
            }
        compound_predicates: (generator) given the compound_length,
            this combines the predicates from simple_cover into
            combinations.
            Let n = len(simple_cover)
                k = compound_length
                L = number of compound_predicates
            Then L = n C k = n! / (n-k)!k!

        predicates: (set)[dudupe.predicates class]

        Args:
            distances: TODO
            candidates: TODO
            data: TODO
            original_length: (int) TODO
            index_include: TODO

        Fields:
            comparison_count: (dict) {
                key: (dedupe.predicates class)
                value: (float)
            }
        """
        self.distances = distances
        self.candidates = candidates
        self.compound_length = 2
        self._old_dupes = []
        self.current_predicates = ()
        self._cached_labels = None

        index_data = Sample(data, 50000, original_length)
        self.sampled_records = Sample(index_data, 2000, original_length)
        predicates = self.distances.predicates()
        self.r = self.compute_r()
        self.fingerprinter = blocking.Fingerprinter(predicates)
        self.fingerprinter.index_all(index_data)
        self.comparison_count = self.compute_comparison_count()

        examples_to_index = candidates.copy()
        if index_include:
            examples_to_index += index_include

        self._index_predicates(examples_to_index)

    @property
    def comparison_count(self):
        return self._comparison_count

    @comparison_count.setter
    def comparison_count(self, comparison_count):
        self._comparison_count = comparison_count

    def compute_comparison_count(self):
        simple_cover = self.covered_pairs(self.fingerprinter, self.sampled_records)
        compound_predicates = self.compound(simple_cover, self.compound_length)
        return self.comparisons(compound_predicates, simple_cover)

    def compute_r(self):
        N = self.sampled_records.original_length
        N_s = len(self.sampled_records)
        r = (N * (N - 1)) / (N_s * (N_s - 1))
        return r

    def fit_transform(self, pairs, y):
        dupes = [pair for label, pair in zip(y, pairs) if label]

        new_dupes = [pair for pair in dupes if pair not in self._old_dupes]
        new_uncovered = (not all(self.predict(new_dupes)))

        if new_uncovered:
            self.current_predicates = self.learn(dupes, recall=1.0)
            self._cached_labels = None
            self._old_dupes = dupes

    def _index_predicates(self, candidates):

        fingerprinter = self.fingerprinter

        records = core.unique((record for pair in candidates for record in pair))

        for field in fingerprinter.index_fields:
            unique_fields = {record[field] for record in records}
            fingerprinter.index(unique_fields, field)

        for pred in fingerprinter.index_predicates:
            pred.freeze(records)

    def _remove(self, index):
        if self._cached_labels is not None:
            self._cached_labels = numpy.delete(self._cached_labels,
                                               index,
                                               axis=0)

    def candidate_scores(self):
        if self._cached_labels is None:
            labels = self.predict(self.candidates)
            self._cached_labels = numpy.array(labels).reshape(-1, 1)

        return self._cached_labels

    def predict(self, candidates):
        labels = []
        for record_1, record_2 in candidates:

            for predicate in self.current_predicates:
                keys = predicate(record_1)
                if keys:
                    if set(predicate(record_2, target=True)) & set(keys):
                        labels.append(1)
                        break
            else:
                labels.append(0)

        return labels

    def learn(self, matches, recall):
        """
        Takes in a set of training pairs and predicates and tries to find
        a good set of blocking rules. Returns a subset of the initial
        predicates which represent the minimum predicates required to
        cover the training data.

            comparison_count: (dict) {
                key: (dedupe.predicates class)
                value: (float)
            }

        Args:
            :matches: (list)[tuple][dict] list of pairs of records which
                are labelled as matches (duplicates) from the active labelling
                [
                    (record1, record2)
                ]
            :recall: (float) number between 0 and 1, minimum fraction of the
                active labelling training data that must be covered by
                the blocking predicate rules

        Returns:
            :final_predicates: (tuple)[tuple] tuple of final predicate rules
                (
                    (predicate1, predicate2, ... predicateN),
                    (predicate3, predicate4),
                    ...
                )
                Each element in final_predicates consists of a tuple of
                N predicates.
        """
        comparison_count = self.comparison_count
        logger.debug(f"Number of initial predicates: {len(self.fingerprinter.predicates)}")
        dupe_cover = Cover(self.fingerprinter.predicates, matches)
        dupe_cover.compound(compound_length=self.compound_length)
        dupe_cover.intersection_update(comparison_count)

        dupe_cover.dominators(cost=comparison_count)

        coverable_dupes = set.union(*dupe_cover.values())
        # logger.debug(dupe_cover.values())
        uncoverable_dupes = [pair for i, pair in enumerate(matches)
                             if i not in coverable_dupes]
        logger.debug(f"Uncoverable dupes: {uncoverable_dupes}")
        epsilon = int((1.0 - recall) * len(matches))
        logger.debug(f"Recall: {recall}, epsilon: {epsilon}")

        if len(uncoverable_dupes) > epsilon:
            logger.warning(OUT_OF_PREDICATES_WARNING)
            logger.debug(uncoverable_dupes)
            epsilon = 0
        else:
            epsilon -= len(uncoverable_dupes)

        for pred in dupe_cover:
            pred.count = comparison_count[pred]
        logger.debug(f"Target: {len(coverable_dupes)-epsilon}")
        searcher = BranchBound(target=len(coverable_dupes) - epsilon,
                               max_calls=2500)
        final_predicates = searcher.search(dupe_cover)
        logger.info('Final predicate set:')
        for predicate in final_predicates:
            logger.info(predicate)
        logger.debug(f"Final predicates: {final_predicates}")
        logger.debug(f"Number of final predicate rules: {len(final_predicates)}")
        return final_predicates

    @staticmethod
    def covered_pairs(fingerprinter, records):
        """

        For each field, there are one or more predicates. A predicate is a class
        defined in dedupe.predicates.py. A predicate is defined by the field
        it is associated with, and the predicate type. A predicate is callable
        (see the __call__ function).

        Pseudo-Algorithm:

            For each predicate, loop through the records list.
            Call the predicate function on each record.

        Args:
            fingerprinter: (blocking.Fingerprinter)
            records: (dict)[dict] Records dictionary

        Returns:
            cover: (dict) {
                key: (dedupe.predicates class)
                value: (dedupe.training.Counter)
            }
        """
        cover = {}

        pair_enumerator = core.Enumerator()
        n_records = len(records)
        logger.info(f"fingerprint predicates: {len(fingerprinter.predicates)}")
        for predicate in fingerprinter.predicates:
            # logger.debug(predicate)
            pred_cover = collections.defaultdict(set)
            for id, record in records.items():
                blocks = predicate(record)
                for block in blocks:
                    pred_cover[block].add(id)

            if not pred_cover:
                continue

            max_cover = max(len(v) for v in pred_cover.values())
            if max_cover == n_records:
                continue

            pairs = (pair_enumerator[pair]
                     for block in pred_cover.values()
                     for pair in itertools.combinations(sorted(block), 2))
            cover[predicate] = Counter(pairs)
            # logger.debug(cover[predicate])
        # logger.debug(len(cover))
        return cover

    def estimate(self, comparisons):
        # Result due to Stefano Allesina and Jacopo Grilli,
        # details forthcoming
        #
        # This estimates the total number of comparisons a blocking
        # rule will produce.
        #
        # While it is true that if we block together records 1 and 2 together
        # N times we have to pay the overhead of that blocking and
        # and there is some cost to each one of those N comparisons,
        # we are using a redundant-free scheme so we only make one
        # truly expensive computation for every record pair.
        #
        # So, how can we estimate how many expensive comparison a
        # predicate will lead to? In other words, how many unique record
        # pairs will be covered by a predicate?

        return self.r * comparisons.total

    def compound(self, simple_predicates, compound_length):
        simple_predicates = sorted(simple_predicates, key=str)

        for pred in simple_predicates:
            yield pred

        CP = predicates.CompoundPredicate

        for i in range(2, compound_length + 1):
            compound_predicates = itertools.combinations(simple_predicates, i)
            for pred_a, pred_b in compound_predicates:
                if pred_a.compounds_with(pred_b) and pred_b.compounds_with(pred_a):
                    yield CP((pred_a, pred_b))

    def comparisons(self, predicates, simple_cover):
        """

        Args:
            simple_cover: (dict) {
                key: (dedupe.predicates class)
                value: (dedupe.training.Counter)
                }
            predicates: (generator)[dedupe.predicates class]

        Returns:
            comparison_count: (dict) {
                key: (dedupe.predicates class)
                value: (float)
                }
        """
        logger.debug("training.BlockLearner.comparisons")
        compounder = self.Compounder(simple_cover)
        comparison_count = {}

        for pred in predicates:
            if len(pred) > 1:
                estimate = self.estimate(compounder(pred))
            else:
                estimate = self.estimate(simple_cover[pred])

            comparison_count[pred] = estimate
        logger.debug(f"Comparison count: {len(comparison_count)}")
        return comparison_count

    class Compounder(object):
        def __init__(self, cover):
            self.cover = cover
            self._cached_predicate = None
            self._cached_cover = None

        def __call__(self, compound_predicate):
            a, b = compound_predicate[:-1], compound_predicate[-1]

            if len(a) > 1:
                if a == self._cached_predicate:
                    a_cover = self._cached_cover
                else:
                    a_cover = self._cached_cover = self(a)
                    self._cached_predicate = a
            else:
                a, = a
                a_cover = self.cover[a]

            return a_cover * self.cover[b]


class Counter(object):
    def __init__(self, iterable):
        if isinstance(iterable, Mapping):
            self._d = iterable
        else:
            d = collections.defaultdict(int)
            for elem in iterable:
                d[elem] += 1
            self._d = d

        self.total = sum(self._d.values())

    def __le__(self, other):
        return (self._d.keys() <= other._d.keys() and
                self.total <= other.total)

    def __eq__(self, other):
        return self._d == other._d

    def __len__(self):
        return len(self._d)

    def __mul__(self, other):

        if len(self) <= len(other):
            smaller, larger = self._d, other._d
        else:
            smaller, larger = other._d, self._d

        # it's meaningfully faster to check in the key dictview
        # of 'larger' than in the dict directly
        larger_keys = larger.keys()

        common = {k: v * larger[k]
                  for k, v in smaller.items()
                  if k in larger_keys}

        return Counter(common)


class BranchBound(object):
    def __init__(self, target, max_calls):
        """
        Args:
            :target: (float) desired number of active label training
                record matches to be covered by the predicate rules
                (computed from recall)
            :max_calls: (int) maximum number of iterations of the search
                function recursion
        """
        self.calls = max_calls
        self.target = target
        self.cheapest_score = float('inf')
        self.original_cover = None

    def search(self, candidates, partial=()):
        # logger.debug("training.BranchBound.search")
        if self.calls <= 0:
            return self.cheapest

        if self.original_cover is None:
            self.original_cover = candidates.copy()
            self.cheapest = candidates

        self.calls -= 1

        covered = self.covered(partial)
        score = self.score(partial)
        if covered >= self.target:
            logger.debug(f"""Number covered >= desired number covered,
                            covered={covered}, target={self.target},
                            score={score}
                            """)
            if score < self.cheapest_score:
                logger.debug(f'Candidates: {partial}')
                self.cheapest = partial
                self.cheapest_score = score

        else:
            window = self.cheapest_score - score
            # logger.debug(f'Cheapest score: {self.cheapest_score}')
            # logger.debug(f'Score: {score}')

            candidates = {p: cover
                          for p, cover in candidates.items()
                          if p.count < window}
            # logger.debug(f"candidates: {candidates}")
            reachable = self.reachable(candidates) + covered

            if candidates and reachable >= self.target:

                order_by = functools.partial(self.order_by, candidates)

                best = max(candidates, key=order_by)

                remaining = self.uncovered_by(candidates,
                                              candidates[best])
                self.search(remaining, partial + (best,))
                del remaining

                reduced = self.remove_dominated(candidates, best)
                self.search(reduced, partial)
                del reduced

        # logger.debug(f"Cheapest final: {self.cheapest}")
        return self.cheapest

    @staticmethod
    def order_by(candidates, p):
        return (len(candidates[p]), -p.count)

    @staticmethod
    def score(partial):
        """
        Args:
            :partial: (tuple)[predicates.CompoundPredicate]
        """
        for p in partial:
            pass
            # logger.debug(f"p: {p}")
            # logger.debug(type(p))
        return sum(p.count for p in partial)

    def covered(self, partial):
        if partial:
            return len(set.union(*(self.original_cover[p]
                                   for p in partial)))
        else:
            return 0

    @staticmethod
    def reachable(dupe_cover):
        if dupe_cover:
            return len(set.union(*dupe_cover.values()))
        else:
            return 0

    @staticmethod
    def remove_dominated(coverage, dominator):
        dominant_cover = coverage[dominator]

        for pred, cover in coverage.copy().items():
            if (dominator.count <= pred.count and
                    dominant_cover >= cover):
                del coverage[pred]

        return coverage

    @staticmethod
    def uncovered_by(coverage, covered):
        remaining = {}
        for predicate, uncovered in coverage.items():
            still_uncovered = uncovered - covered
            if still_uncovered:
                remaining[predicate] = still_uncovered

        return remaining


class Cover(object):
    def __init__(self, *args):
        if len(args) == 1:
            self._d, = args
        else:
            self._d = {}
            predicates, pairs = args
            self._cover(predicates, pairs)

    def __repr__(self):
        return 'Cover:' + str(self._d.keys())

    def _cover(self, predicates, pairs):
        for predicate in predicates:
            coverage = {i for i, (record_1, record_2)
                        in enumerate(pairs)
                        if (set(predicate(record_1)) &
                            set(predicate(record_2, target=True)))}
            if coverage:
                self._d[predicate] = coverage

    def compound(self, compound_length):
        simple_predicates = sorted(self._d, key=str)
        CP = predicates.CompoundPredicate

        for i in range(2, compound_length + 1):
            compound_predicates = itertools.combinations(simple_predicates, i)

            for compound_predicate in compound_predicates:
                a, b = compound_predicate[:-1], compound_predicate[-1]
                if len(a) == 1:
                    a = a[0]

                if a in self._d:
                    compound_cover = self._d[a] & self._d[b]
                    if compound_cover:
                        self._d[CP(compound_predicate)] = compound_cover

    def dominators(self, cost):
        """
        candidate_match: list of active label training ids which are covered
            by the compound predicate rule
        candidate_cost: (float) computational cost of this predicate

        Pseudo-Algorithm
        1. Loop through list of predicates
            a. Loop through remainder of list of predicates
                - If a better or equally good match later in the list
                    is found, continue to the next predicate in outer
                    loop
                - A better match is one with lower computational cost
                    and one which covers more records in training data
                - If not, add the predicate to the dominants list
        """
        logger.debug("training.Cover.dominators")

        def sort_key(x):
            return (-cost[x], len(self._d[x]))

        ordered_predicates = sorted(self._d, key=sort_key)
        dominants = {}
        for i, candidate in enumerate(ordered_predicates):
            candidate_match = self._d[candidate]
            candidate_cost = cost[candidate]
            for pred in ordered_predicates[(i + 1):]:
                other_match = self._d[pred]
                other_cost = cost[pred]
                better_or_equal = (other_match >= candidate_match and
                                   other_cost <= candidate_cost)
                if better_or_equal:
                    break
            else:
                dominants[candidate] = candidate_match
        # logger.debug(f"dominants: {dominants}")
        self._d = dominants

    def __iter__(self):
        return iter(self._d)

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

    def __getitem__(self, k):
        return self._d[k]

    def copy(self):
        return Cover(self._d.copy())

    def update(self, *args, **kwargs):
        self._d.update(*args, **kwargs)

    def __eq__(self, other):
        return self._d == other._d

    def intersection_update(self, other):
        self._d = {k: self._d[k] for k in set(self._d) & set(other)}


class Sample(dict):

    def __init__(self, d, sample_size, original_length):
        """
        Args:
            original_length: (int)
        """
        if len(d) <= sample_size:
            super().__init__(d)
        else:
            _keys = tuple(d.keys())
            sample = (random.choice(_keys) for _ in range(sample_size))
            super().__init__({k: d[k] for k in sample})
        if original_length is None:
            self.original_length = len(d)
        else:
            self.original_length = original_length


class DedupeSampler(object):

    def __init__(self, distances):
        self.distances = distances

    def sample(self, data, blocked_proportion, sample_size):
        blocked_sample_size = int(blocked_proportion * sample_size)
        predicates = list(self.distances.predicates(index_predicates=False))

        data = sampling.randomDeque(data)
        blocked_sample_keys = sampling.dedupeBlockedSample(blocked_sample_size,
                                                           predicates,
                                                           data)

        random_sample_size = sample_size - len(blocked_sample_keys)
        random_sample_keys = set(core.randomPairs(len(data),
                                                  random_sample_size))
        data = dict(data)

        return [(data[k1], data[k2])
                for k1, k2
                in blocked_sample_keys | random_sample_keys]


OUT_OF_PREDICATES_WARNING = """Ran out of predicates: Dedupe tries to find blocking rules
    that will work well with your data. Sometimes it can't find great ones, and you'll
    get this warning. It means that there are some pairs of true records that
    dedupe may never compare. If you are getting bad results, try increasing the `max_comparison`
    argument to the train method"""
