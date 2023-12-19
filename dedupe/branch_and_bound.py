from __future__ import annotations

import functools
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
    return {
        pred: still_uncovered
        for pred, uncovered in coverage.items()
        if (still_uncovered := uncovered - covered)
    }


def _order_by(
    candidates: Mapping[Predicate, Sequence[Any]], p: Predicate
) -> tuple[int, float]:
    return (len(candidates[p]), -p.cover_count)


def _score(partial: Iterable[Predicate]) -> float:
    return sum(p.cover_count for p in partial)


def search(original_cover: Cover, target: int, calls: int) -> Partial:
    def _covered(partial: Partial) -> int:
        return (
            len(frozenset.union(*(original_cover[p] for p in partial)))
            if partial
            else 0
        )

    cheapest_score = float("inf")
    cheapest: Partial = ()

    start: tuple[Cover, Partial] = (original_cover, ())
    to_explore = [start]

    while to_explore and calls:
        candidates, partial = to_explore.pop()

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

                reduced = _remove_dominated(candidates, best)
                to_explore.append((reduced, partial))

                remaining = _uncovered_by(candidates, candidates[best])
                to_explore.append((remaining, partial + (best,)))

        elif score < cheapest_score:
            cheapest = partial
            cheapest_score = score

        calls -= 1

    return cheapest
