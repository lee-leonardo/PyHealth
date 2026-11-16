# Author: lee-leonardo
# Description: Runnable KEEP co-occurrence matrix example on synthetic OMOP
#   data. Demonstrates the real PyHealth path (OMOPDataset.iter_patients() ->
#   ICD mapper -> build_keep_omop_ancestor_rollup -> build_keep_cooccurrence)
#   without any data credentials.

"""Build the KEEP co-occurrence matrix on a synthetic OMOP cohort.

KEEP (Elhussein et al., CHIL 2025, https://arxiv.org/abs/2510.05049,
appendix A.4) learns concept embeddings from a patient co-occurrence matrix
built over a SNOMED condition hierarchy. This script demonstrates the
PyHealth path end to end on fully synthetic inputs, so it needs no data
credentials:

1. a minimal OMOP CDM cohort (person, visit_occurrence, death and
   condition_occurrence CSV files) whose condition rows follow the peer
   converter's conventions: one ``condition_occurrence_id`` per diagnosis
   row, ``condition_concept_id`` 0 beside an ``ICD9CM:code`` or
   ``ICD10CM:code`` ``condition_source_value`` for unmapped ICD rows;
2. a synthetic Athena vocabulary bundle (CONCEPT, CONCEPT_RELATIONSHIP,
   CONCEPT_ANCESTOR) with the Disease root 4274025, a depth-6 deep source,
   a multi-parent concept, an isolated retained orphan, and one
   standard-stitched closure edge;
3. the real readers: ``OMOPDataset.iter_patients()`` for the histories,
   ``OMOPVocabulary.icd_mapper()`` for the ICD codes and
   ``build_keep_omop_ancestor_rollup`` for the roll-up, configured with
   the KEEP profile (minimum root separations from the OMOP closure,
   ``max_depth=5``, ``target_policy="all_retained"``,
   ``path_source="closure"``, ``include_root=False``,
   ``on_unmapped="empty"``);
4. :func:`pyhealth.tasks.build_keep_cooccurrence` over the whole cohort,
   whose axes share the processor's retained vocabulary, so the retained
   graph and later embeddings can use the same concept index.

The paper describes whole histories, a two-occurrence presence rule,
multi-parent roll-up and a vocabulary shared with the Node2Vec graph. The
diagnosis-row identity, the post-roll-up thresholding, the prevalence
diagonal and the mirrored pair counts are this project's choices, recorded
in the builder's documentation. Embedding training and clinical evaluation
are downstream. For later held-out evaluation, build the matrix using the
histories selected for embedding fitting, and keep the returned
``concept_ids`` order as the embedding vocabulary.

Usage:
    python examples/omop_keep_cooccurrence_matrix.py [--workdir DIR]

Without ``--workdir`` the synthetic inputs go to a temporary folder that is
deleted afterwards; pass ``--workdir`` to keep them for inspection.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from pyhealth.datasets import OMOPDataset
from pyhealth.graph.cooccurrence import build_cooccurrence
from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import (
    KeepCooccurrenceResult,
    build_keep_cooccurrence,
    build_keep_omop_ancestor_rollup,
)

#: OMOP concept_id of SNOMED 64572001 "Disease", the root KEEP uses.
ROOT = 4274025

# concept_id, concept_name, domain_id, vocabulary_id, standard_concept,
# concept_code (every concept valid; standard "" marks nonstandard)
CONCEPTS = [
    (ROOT, "Disease", "Condition", "SNOMED", "S", "64572001"),
    (100, "Heart disease", "Condition", "SNOMED", "S", "56265001"),
    (111, "Spinal disorder", "Condition", "SNOMED", "S", "1000111"),
    (101, "Ischemic heart disease", "Condition", "SNOMED", "S", "414545008"),
    (104, "Two-parent disorder", "Condition", "SNOMED", "S", "1000004"),
    (102, "Acute MI", "Condition", "SNOMED", "S", "57054005"),
    (108, "Late complication", "Condition", "SNOMED", "S", "1000108"),
    (109, "Boundary disorder", "Condition", "SNOMED", "S", "1000109"),
    (110, "Very deep disorder", "Condition", "SNOMED", "S", "1000110"),
    (600, "Cardiac observation", "Observation", "SNOMED", "S", "1000600"),
    (105, "Orphan disorder", "Condition", "SNOMED", "S", "1000005"),
    (222, "Cord disorder", "Condition", "SNOMED", "S", "1000222"),
    (333, "Cord compression", "Condition", "SNOMED", "S", "1000333"),
    (444, "Myelopathy precursor", "Condition", "SNOMED", "S", "1000444"),
    (140708, "Myelopathy due to another disorder", "Condition", "SNOMED",
     "S", "100140708"),
    (4105351, "Nonstandard intermediate", "Condition", "SNOMED", "",
     "1004105351"),
    (250, "Stitched disorder", "Condition", "SNOMED", "S", "1000250"),
    (9001, "Convergent ICD-10 K1", "Condition", "ICD10CM", "", "K1"),
    (9002, "ICD-10 I214", "Condition", "ICD10CM", "", "I214"),
    (9003, "ICD-9 0010", "Condition", "ICD9CM", "", "0010"),
    (9004, "ICD-10 Z999 (no mapping)", "Condition", "ICD10CM", "", "Z999"),
    (9006, "ICD-10 I250", "Condition", "ICD10CM", "", "I250"),
]

# child, parent, relationship_id ("Is a" hierarchy or "Maps to" ICD links)
RELATIONSHIPS = [
    (100, ROOT, "Is a"),
    (111, ROOT, "Is a"),
    (101, 100, "Is a"),
    (104, 100, "Is a"),
    (104, 111, "Is a"),
    (102, 101, "Is a"),
    (108, 102, "Is a"),
    (109, 108, "Is a"),
    (110, 109, "Is a"),
    (600, 100, "Is a"),
    (105, 600, "Is a"),
    (222, 111, "Is a"),
    (333, 222, "Is a"),
    (444, 333, "Is a"),
    (140708, 444, "Is a"),
    (4105351, 140708, "Is a"),
    (250, 4105351, "Is a"),
    (9001, 110, "Maps to"),
    (9001, 104, "Maps to"),
    (9002, 110, "Maps to"),
    (9003, 109, "Maps to"),
    (9006, 250, "Maps to"),
]

# Closure rows the "Is a" adjacency between kept concepts cannot reproduce:
# the standard-stitched ancestry of 250, whose only "Is a" route runs
# through the nonstandard 4105351 (the Athena discrepancy of the standalone
# rehearsal). 250 sits at closure depth 6, so it is a deep source that only
# path_source="closure" can roll up.
STITCHED_ANCESTOR_ROWS = [
    (ROOT, 250, 6, 6),
    (111, 250, 5, 5),
    (222, 250, 4, 4),
    (333, 250, 3, 3),
    (444, 250, 2, 2),
    (140708, 250, 1, 1),
]

# The OMOP config the cohort needs: condition occurrences must expose the
# diagnosis id, the canonical concept id, the ICD source value and the
# visit id. The default OMOP config omits the source value and the
# diagnosis id; the peer's converter branch adds only the source value.
OMOP_CONFIG = """\
version: "5.3"
tables:
  person:
    file_path: "person.csv"
    patient_id: "person_id"
    timestamp: null
    timestamp_format: "%Y-%m-%d %H:%M:%S"
    attributes:
      - "year_of_birth"
      - "month_of_birth"
      - "day_of_birth"
  visit_occurrence:
    file_path: "visit_occurrence.csv"
    patient_id: "person_id"
    timestamp: "visit_start_datetime"
    timestamp_format: "%Y-%m-%d %H:%M:%S"
    attributes:
      - "visit_occurrence_id"
      - "visit_start_datetime"
      - "visit_end_datetime"
  death:
    file_path: "death.csv"
    patient_id: "person_id"
    timestamp: "death_datetime"
    timestamp_format: "%Y-%m-%d %H:%M:%S"
    attributes:
      - "death_datetime"
  condition_occurrence:
    file_path: "condition_occurrence.csv"
    patient_id: "person_id"
    timestamp: "condition_start_datetime"
    timestamp_format: "%Y-%m-%d %H:%M:%S"
    attributes:
      - "condition_occurrence_id"
      - "condition_concept_id"
      - "condition_source_value"
      - "visit_occurrence_id"
"""

# person_id, year of birth
PERSONS = [(1, 1950), (2, 1960), (3, 1970), (4, 1980), (5, 1990)]

# visit_occurrence_id, person_id, visit_start/end datetime
VISITS = [
    (11, 1, "2118-01-05 09:00:00", "2118-01-08 09:00:00"),
    (12, 1, "2118-06-10 10:00:00", "2118-06-11 10:00:00"),
    (21, 2, "2118-02-01 08:00:00", "2118-02-02 08:00:00"),
    (31, 3, "2118-03-11 07:00:00", "2118-03-15 07:00:00"),
    (41, 4, "2118-04-20 06:00:00", "2118-04-21 06:00:00"),
    (51, 5, "2118-05-25 05:00:00", "2118-05-26 05:00:00"),
]

# condition_occurrence_id, person_id, condition_concept_id (0 = ICD source
# value), condition_start_datetime, visit_occurrence_id (empty = none),
# condition_source_value
CONDITIONS = [
    (201, 1, "110", "2118-01-05 09:30:00", "11", ""),
    (202, 1, "0", "2118-01-06 10:30:00", "11", "ICD10CM:K1"),
    (203, 1, "0", "2118-06-10 11:00:00", "12", "ICD9CM:0010"),
    (301, 2, "0", "2118-02-01 08:30:00", "21", "ICD10CM:K1"),
    (401, 3, "110", "2118-03-11 07:30:00", "31", ""),
    (402, 3, "110", "2118-03-12 07:30:00", "31", ""),
    (403, 3, "110", "2118-03-13 07:30:00", "31", ""),
    (501, 4, "0", "2118-04-20 06:30:00", "41", "ICD10CM:I250"),
    (502, 4, "0", "2118-04-20 06:35:00", "41", "ICD10CM:I250"),
    (601, 5, "", "2118-05-25 05:30:00", "51", ""),
    (602, 5, "0", "2118-05-25 05:35:00", "51", "ICD10CM:Z999"),
]


def _write(path, header, rows, separator="\t") -> None:
    """Writes one delimited table (Athena files are tab-separated)."""
    lines = [separator.join(header)]
    lines += [separator.join(str(value) for value in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concept_ancestor_rows():
    """Derives CONCEPT_ANCESTOR the way Athena does, then appends stitches.

    Standard "Is a" links between valid standard concepts produce every
    (ancestor, descendant, min, max) closure row, including self rows at
    zero; the stitched rows above are appended afterwards.
    """
    standard = {row[0] for row in CONCEPTS if row[4] == "S"}
    parents = defaultdict(set)
    for child, parent, relation in RELATIONSHIPS:
        if relation == "Is a" and {child, parent} <= standard:
            parents[child].add(parent)

    def levels(concept):
        found = defaultdict(list)
        stack = [(concept, 0)]
        while stack:
            node, distance = stack.pop()
            found[node].append(distance)
            for parent in parents[node]:
                stack.append((parent, distance + 1))
        return found

    rows = []
    for concept in sorted(standard):
        for ancestor, distances in levels(concept).items():
            rows.append((ancestor, concept, min(distances), max(distances)))
    return rows + STITCHED_ANCESTOR_ROWS


def write_athena_bundle(folder) -> None:
    """Writes the synthetic Athena vocabulary tables."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write(
        folder / "CONCEPT.csv",
        [
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "concept_class_id", "standard_concept", "concept_code",
            "invalid_reason",
        ],
        [
            (cid, name, domain, vocab, "Clinical Finding", std, code, "")
            for cid, name, domain, vocab, std, code in CONCEPTS
        ],
    )
    _write(
        folder / "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        [(child, parent, relation, "") for child, parent, relation
         in RELATIONSHIPS],
    )
    _write(
        folder / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id", "descendant_concept_id",
            "min_levels_of_separation", "max_levels_of_separation",
        ],
        _concept_ancestor_rows(),
    )


def write_omop_cohort(folder) -> Path:
    """Writes the OMOP CDM CSVs and the config; returns the config path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write(
        folder / "person.csv",
        ["person_id", "year_of_birth", "month_of_birth", "day_of_birth"],
        [(person_id, year, 1, 1) for person_id, year in PERSONS],
        separator=",",
    )
    _write(
        folder / "visit_occurrence.csv",
        [
            "visit_occurrence_id", "person_id", "visit_start_datetime",
            "visit_end_datetime",
        ],
        VISITS,
        separator=",",
    )
    _write(
        folder / "death.csv",
        ["person_id", "death_datetime"],
        [],  # no deaths in the synthetic cohort
        separator=",",
    )
    _write(
        folder / "condition_occurrence.csv",
        [
            "condition_occurrence_id", "person_id", "condition_concept_id",
            "condition_start_datetime", "visit_occurrence_id",
            "condition_source_value",
        ],
        CONDITIONS,
        separator=",",
    )
    config_path = folder / "omop_condition_ids.yaml"
    config_path.write_text(OMOP_CONFIG, encoding="utf-8")
    return config_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--workdir",
        help="Keep the synthetic inputs and cache in this folder",
        default=None,
    )
    return parser.parse_args(argv)


def build_matrix(
    athena_dir, omop_dir, config_path, cache_dir
) -> KeepCooccurrenceResult:
    """Runs the documented OMOP -> ICD -> roll-up -> matrix path."""
    vocabulary = OMOPVocabulary(athena_dir)
    processor = build_keep_omop_ancestor_rollup(
        vocabulary,
        target_policy="all_retained",
        include_root=False,
        max_depth=5,
        path_source="closure",
        on_unmapped="empty",
    )
    mapper = vocabulary.icd_mapper()
    dataset = OMOPDataset(
        root=str(omop_dir),
        tables=["condition_occurrence"],
        config_path=str(config_path),
        cache_dir=str(cache_dir),
    )
    concept_ids = sorted(processor.retained_codes)
    return build_keep_cooccurrence(
        dataset.iter_patients(),
        rollup=processor,
        icd_mapper=mapper,
        concept_ids=concept_ids,
        matrix_builder=lambda rows, *, concept_ids: build_cooccurrence(
            rows, concept_ids=concept_ids
        ),
    )


def report(result: KeepCooccurrenceResult) -> None:
    """Prints the matrix and the aggregate counts (synthetic data only)."""
    print(f"Retained vocabulary: {len(result.concept_ids)} concepts")
    print(f"Patients processed: {result.patients_processed}")
    print(
        "Patients with a present condition: "
        f"{result.patients_with_present_targets}"
    )
    print(f"Unique diagnosis events: {result.unique_diagnosis_events}")
    print(f"Mapped events: {result.mapped_events}")
    for reason, count in result.unmapped_events.items():
        print(f"Unmapped events ({reason}): {count}")
    print("Co-occurrence matrix (patient counts; axes = concept ids):")
    dense = result.matrix.to_dense().tolist()
    for code, row in zip(result.concept_ids, dense):
        print(f"  {code}: {row}")


def main(argv=None) -> KeepCooccurrenceResult:
    args = parse_args(argv)
    if args.workdir:
        work = Path(args.workdir)
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="keep_cooccurrence_"))
        cleanup = True
    try:
        athena_dir = work / "athena"
        omop_dir = work / "omop"
        write_athena_bundle(athena_dir)
        config_path = write_omop_cohort(omop_dir)
        result = build_matrix(athena_dir, omop_dir, config_path, work / "cache")
        report(result)
        return result
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
