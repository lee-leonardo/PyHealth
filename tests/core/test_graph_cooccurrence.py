"""Counts preselected patient concepts without a task or hierarchy."""

import pytest
import torch

from pyhealth.graph.cooccurrence import build_cooccurrence


def test_patient_counts_keep_axis_order_and_isolated_concepts():
    matrix = build_cooccurrence(
        iter([{"A", "B"}, {"A"}, set()]), concept_ids=("B", "Z", "A")
    )

    assert matrix.to_dense().tolist() == [[1, 0, 1], [0, 0, 0], [1, 0, 2]]
    assert matrix.layout == torch.sparse_coo
    assert matrix.dtype == torch.int64
    assert matrix.device.type == "cpu"
    assert matrix.is_coalesced()


@pytest.mark.parametrize("concept_ids", [(), ("B", "Z", "A")])
def test_empty_cohort_keeps_the_axis_shape(concept_ids):
    matrix = build_cooccurrence(iter(()), concept_ids=concept_ids)

    assert matrix.shape == (len(concept_ids), len(concept_ids))
    assert matrix._nnz() == 0


def test_rejects_concept_absent_from_axes():
    with pytest.raises(ValueError, match="is not part of concept_ids"):
        build_cooccurrence([{"unknown"}], concept_ids=("A",))
