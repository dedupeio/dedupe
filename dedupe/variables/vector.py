from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from dedupe import predicates
from dedupe.variables.base import FieldType

if TYPE_CHECKING:
    from dedupe._typing import PredicateFunction

logger = logging.getLogger(__name__)


def cosine_distance(vec1: Any, vec2: Any) -> float:
    """
    Computes cosine distance between two 1-D numeric vector representations.

    Returns:
        float in range [0.0, 1.0], where 0.0 indicates collinearity (identical direction),
        and 1.0 indicates orthogonality or opposite direction.
        Returns np.nan if either vector is None, empty, or mismatched dimension.
    """
    if vec1 is None or vec2 is None:
        return float(np.nan)

    try:
        u = np.asarray(vec1, dtype=np.float32).ravel()
        v = np.asarray(vec2, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return float(np.nan)

    if u.size == 0 or v.size == 0 or u.shape != v.shape:
        return float(np.nan)

    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u == 0.0 or norm_v == 0.0:
        return 1.0

    cos_sim = float(np.dot(u, v) / (norm_u * norm_v))
    # Numerical stability clamp
    cos_sim = max(-1.0, min(1.0, cos_sim))

    # Scale from [-1, 1] similarity to [0, 1] normalized distance
    return float(0.5 * (1.0 - cos_sim))


def euclidean_distance(vec1: Any, vec2: Any) -> float:
    """
    Computes Euclidean distance between two numeric vector representations.
    """
    if vec1 is None or vec2 is None:
        return float(np.nan)

    try:
        u = np.asarray(vec1, dtype=np.float32).ravel()
        v = np.asarray(vec2, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return float(np.nan)

    if u.size == 0 or v.size == 0 or u.shape != v.shape:
        return float(np.nan)

    return float(np.linalg.norm(u - v))


def vector_sign_hash_predicate(field: Any) -> tuple[str, ...]:
    """
    Locality-Sensitive Hashing (LSH) sign bit predicate for continuous vectors.
    Partitions the vector into bit signatures across coordinate blocks.
    """
    if field is None:
        return ()

    try:
        vec = np.asarray(field, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return ()

    if vec.size == 0:
        return ()

    # Hyperplane sign hashing across sliced coordinate partitions
    num_bands = 8
    band_size = max(1, len(vec) // num_bands)
    predicates_list = []

    for b in range(num_bands):
        start = b * band_size
        end = min(len(vec), start + band_size)
        if start >= len(vec):
            break
        chunk = vec[start:end]
        # Generate compact bitstring of signs (> 0)
        bits = "".join("1" if x > 0 else "0" for x in chunk[:16])
        predicates_list.append(f"vband_{b}_{bits}")

    return tuple(predicates_list)


class VectorType(FieldType):
    """
    Variable type for dense numerical embeddings (Sentence-Transformers, OpenAI, etc.).
    Supports cosine distance and euclidean distance comparators, with LSH sign predicates.
    """

    type = "Vector"
    _Predicate = predicates.SimplePredicate
    _predicate_functions: Sequence[PredicateFunction] = (vector_sign_hash_predicate,)

    def __init__(
        self,
        field: str,
        metric: str = "cosine",
        name: str | None = None,
        has_missing: bool = False,
    ):
        super().__init__(field, name=name, has_missing=has_missing)
        self.metric = metric

        if metric == "cosine":
            self.comparator = cosine_distance
        elif metric == "euclidean":
            self.comparator = euclidean_distance
        else:
            raise ValueError(f"Unsupported vector metric '{metric}'. Expected 'cosine' or 'euclidean'.")
