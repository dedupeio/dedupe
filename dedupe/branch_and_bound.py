from __future__ import annotations

import functools
import warnings
from typing import Any, Iterable, Mapping, Sequence, Tuple

from ._typing import Cover
from .predicates import Predicate

Partial = Tuple[Predicate, ...]


def _reachable(dupe_cover: Mapping[Any, frozenset[int]]) -> int:
    return len(frozenset.union(*dupe_cover.values())) if dupe_cover else 0


def _remove_dominated(coverage: Cover, dominator: Predicate) -> Cover:
    dominant_cover = coverage[dominator]

    return {
        pred: cover
        for pred, cover in coverage.items()
        if not (dominator.cover_count <= pred.cover_count and dominant_cover >= cover)
    }


def _uncovered_by(
    coverage: Mapping[Any, frozenset[int]], covered: frozenset[int]
) -> dict[Any, frozenset[int]]:
    remaining = {}
    for predicate, uncovered in coverage.items():
        still_uncovered = uncovered - covered
        if still_uncovered:
            remaining[predicate] = still_uncovered

    return remaining


def _order_by(
    candidates: Mapping[Predicate, Sequence[Any]], p: Predicate
) -> tuple[int, float]:
    return (len(candidates[p]), -p.cover_count)


def _score(partial: Iterable[Predicate]) -> float:
    return sum(p.cover_count for p in partial)


def _suppress_recursion_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RecursionError:
            warnings.warn("Recursion limit eached while searching for predicates")

    return wrapper


def search(candidates, target: int, max_calls: int) -> Partial:
    calls = max_calls

    cheapest_score = float("inf")
    cheapest: Partial = ()

    original_cover = candidates.copy()

    def _covered(partial: Partial) -> int:
        return (
            len(frozenset.union(*(original_cover[p] for p in partial)))
            if partial
            else 0
        )

    @_suppress_recursion_error
    def walk(candidates: Cover, partial: Partial = ()) -> None:
        nonlocal calls
        nonlocal cheapest
        nonlocal cheapest_score

        if calls <= 0:
            return

        calls -= 1

        covered = _covered(partial)
        score = _score(partial)

        if covered < target:
            window = cheapest_score - score
            candidates = {
                p: cover for p, cover in candidates.items() if p.cover_count < window
            }

            reachable = _reachable(candidates) + covered

            if candidates and reachable >= target:
                order_by = functools.partial(_order_by, candidates)
                best = max(candidates, key=order_by)

                remaining = _uncovered_by(candidates, candidates[best])
                walk(remaining, partial + (best,))
                del remaining

                reduced = _remove_dominated(candidates, best)
                walk(reduced, partial)
                del reduced

        elif score < cheapest_score:
            cheapest = partial
            cheapest_score = score

    walk(candidates)
    return cheapest
