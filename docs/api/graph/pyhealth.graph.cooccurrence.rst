.. Author: lee-leonardo (NetID: ll37)

.. _patient-cooccurrence:

Patient co-occurrence matrices
==============================

A patient co-occurrence matrix summarizes which concepts appear together
across a cohort. Each patient contributes a set of present concepts.
Entry ``C[i, j]`` counts patients whose sets contain both concepts.
The diagonal ``C[i, i]`` gives the number of patients with concept ``i``.
For example, a pair count of 10 can be interpreted alongside each
concept's prevalence, or used to weight an edge or train concept embeddings.

Choose a single order for the concepts and use it for both axes. A row for a
concept absent from the cohort still has a position in that order. The
matrix is symmetric and uses sparse COO storage, which records the nonzero
counts with their row and column indices. The caller defines patient
presence. For example, the :doc:`KEEP workflow
<../tasks/pyhealth.tasks.keep>` establishes presence after ancestor roll-up.

Counting patient sets
---------------------

This example adapts an in-memory Polars frame with one row per patient.
The first patient contributes one count to A, B, and their pair. The
second contributes one count to A. The third has no present concepts.

.. code-block:: python

    import polars as pl

    from pyhealth.graph.cooccurrence import build_cooccurrence

    patients = pl.DataFrame({"concepts": [["A", "B"], ["A"], []]})
    patient_sets = (set(codes) for (codes,) in patients.select("concepts").iter_rows())
    counts = build_cooccurrence(patient_sets, concept_ids=["B", "Z", "A"])
    counts.to_dense().tolist()
    # [[1, 0, 1], [0, 0, 0], [1, 0, 2]]

An ETL pipeline can also yield patient sets from successive input batches.
Combine a patient's events into one set before yielding it, including when
its events span multiple batches. The builder consumes the iterable once
and keeps a counter for the distinct observed pairs. A patient with
``k`` present concepts contributes ``k * (k + 1) / 2`` unordered
pairs including the diagonal. Pair counting runs on the CPU and uses no
files. Set ``device="cuda:0"`` to construct the sparse tensor on CUDA
when supported by the PyTorch installation.

Transforming the counts
-----------------------

Use ``transform`` to derive weights from the coalesced count tensor and
cohort size. The callable runs once, on the requested device, after
counting. The cohort size includes patients with empty concept sets.

For positive pointwise mutual information (PPMI), compare the observed
joint prevalence with the prevalence expected under independence. For
an observed pair, its weight is
``max(0, log(N * C[i, j] / (C[i, i] * C[j, j])))``, where ``N``
is the cohort size. This example drops weights that clip to zero.

.. code-block:: python

    import torch

    def ppmi(counts, patient_count):
        rows, cols = counts.indices()
        joint = counts.values().to(torch.float64)
        marginal_counts = joint.new_zeros(counts.size(0))
        diagonal = rows == cols
        marginal_counts[rows[diagonal]] = joint[diagonal]
        denominator = marginal_counts[rows] * marginal_counts[cols]
        weights = (patient_count * joint / denominator).log()
        positive = weights > 0
        return torch.sparse_coo_tensor(
            counts.indices()[:, positive],
            weights[positive],
            counts.shape,
            device=counts.device,
            is_coalesced=True,
        )

    weights = build_cooccurrence(
        iter([{"A", "B"}, {"A"}, set()]),
        concept_ids=["A", "B"],
        transform=ppmi,
    )

Cosine similarity between the concepts' binary patient incidence vectors
uses ``C[i, j] / sqrt(C[i, i] * C[j, j])`` on observed pairs. Absent
concepts keep zero rows and columns. A different similarity, such as
cosine similarity between rows of the co-occurrence matrix, would use
those rows and their norms as its inputs.

API Reference
-------------

.. autofunction:: pyhealth.graph.cooccurrence.build_cooccurrence
