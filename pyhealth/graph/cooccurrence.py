# Author: lee-leonardo
# Description: Sparse patient co-occurrence over caller-supplied concept sets.

"""Sparse patient co-occurrence over preselected concepts."""

from collections import Counter
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from itertools import combinations_with_replacement

import torch


def build_cooccurrence(
    patient_concepts: Iterable[AbstractSet[str]], *, concept_ids: Sequence[str]
) -> torch.Tensor:
    """Count each patient's concept pairs in the supplied axis order.

    Each input set represents one patient and may be empty. The diagonal
    counts patient prevalence, and off-diagonal counts are symmetric.
    Returns a coalesced CPU int64 COO tensor with both axes of length
    ``len(concept_ids)``. All axis IDs must be unique and nonempty. Every
    supplied patient concept must be present on the axes.

    Example:
        >>> from pyhealth.graph.cooccurrence import build_cooccurrence
        >>> matrix = build_cooccurrence(
        ...     [{"A", "B"}, {"A"}, set()], concept_ids=["B", "Z", "A"]
        ... )
        >>> matrix.to_dense().tolist()
        [[1, 0, 1], [0, 0, 0], [1, 0, 2]]
    """
    if isinstance(concept_ids, (str, bytes)):
        raise TypeError("concept_ids must be a sequence of strings")
    axis_index: dict[str, int] = {}
    for position, code in enumerate(concept_ids):
        if not isinstance(code, str):
            raise TypeError("concept_ids must contain strings")
        if not code or code in axis_index:
            raise ValueError("concept_ids must contain unique, nonempty IDs")
        axis_index[code] = position

    entries: Counter[tuple[int, int]] = Counter()
    for concepts in patient_concepts:
        try:
            positions = sorted(axis_index[code] for code in concepts)
        except KeyError as error:
            raise ValueError(
                f"patient concept {error.args[0]!r} is not part of concept_ids"
            ) from None
        for row, column in combinations_with_replacement(positions, 2):
            entries[row, column] += 1
            if row != column:
                entries[column, row] += 1

    indices = torch.tensor(list(entries), dtype=torch.int64, device="cpu")
    indices = indices.reshape(-1, 2).T.contiguous()
    values = torch.tensor(list(entries.values()), dtype=torch.int64, device="cpu")
    size = len(axis_index)
    return torch.sparse_coo_tensor(
        indices, values, (size, size), device="cpu"
    ).coalesce()
