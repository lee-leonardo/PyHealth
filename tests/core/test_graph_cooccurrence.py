"""Counts preselected patient concepts without a task or hierarchy."""

from functools import partial

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


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("patients", [({"A", "B"}, {"A"}, set()), ()])
def test_configured_builder_places_counts_and_passes_cohort_size(device, patients):
    calls = []

    def normalize(counts, patient_count):
        assert counts.device == torch.device(device)
        assert counts.is_coalesced()
        assert counts.dtype == torch.int64
        calls.append(patient_count)
        return counts.to(torch.float64) / max(patient_count, 1)

    builder = partial(build_cooccurrence, device=device, transform=normalize)
    matrix = builder(iter(patients), concept_ids=("B", "Z", "A"))
    expected = build_cooccurrence(iter(patients), concept_ids=("B", "Z", "A"))

    assert calls == [len(patients)]
    assert matrix.device == torch.device(device)
    assert matrix.layout == torch.sparse_coo
    torch.testing.assert_close(
        matrix.to_dense().cpu(), expected.to_dense() / max(len(patients), 1),
        check_dtype=False,
    )
