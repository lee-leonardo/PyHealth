"""Tests for the KEEP co-occurrence matrix builder on toy patients.

The toy hierarchy mirrors the OMOP situation the builder targets: a root
``"1"``; retained concepts ``"10"`` (A), ``"11"`` (B) and ``"12"`` (Z, an
isolated concept with a supplied root depth but no edges); and deep sources
``"801"`` (X, under ``"10"``), ``"802"`` (C, under both ``"10"`` and
``"11"``, the multi-parent case) and ``"803"`` (Y, under ``"11"``). With
``max_depth=1`` and ``all_retained`` the retained vocabulary is exactly
``{"10", "11", "12"}`` and every deep source rolls up to it.

The ICD mapper is the peer's real ``ICDToSNOMEDMapper`` over a small
polars table: ``K1`` maps one-to-many to ``801`` and ``802`` (converging
routes), ``K2`` to ``801``, ICD-9 ``0010`` (leading zero) to ``803``,
``UNKN`` to ``999`` (a concept outside the toy universe) and ``OBSX`` to
``90`` (an Observation-domain concept, outside the eligible domain).

Exact fixture (axes B, Z, A = ``("11", "12", "10")``), from the builder
handoff: patient 1 has two different diagnoses across two visits, each
resolving to A and B, with one diagnosis reaching A through multiple ICD
targets; patient 2 has a single such diagnosis (convergence alone must not
establish presence); patient 3 has three distinct A diagnoses in one visit.
Expected matrix ``[[1, 0, 1], [0, 0, 0], [1, 0, 2]]``.
"""

import unittest
from datetime import datetime

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.graph import KnowledgeGraph
from pyhealth.medcode import ICDToSNOMEDMapper
from pyhealth.processors import AncestorRollupProcessor, UnmappedCodeError
from pyhealth.tasks import build_keep_cooccurrence

TIMESTAMP = datetime(2020, 1, 1, 12, 0, 0)

TRIPLES = [
    ("1", "is_a", "10"),
    ("1", "is_a", "11"),
    ("10", "is_a", "801"),
    ("10", "is_a", "802"),
    ("11", "is_a", "802"),
    ("11", "is_a", "803"),
]
ENTITY2ID = {
    "1": 0, "10": 1, "11": 2, "12": 3, "801": 4, "802": 5, "803": 6,
    "90": 7,
}
ROOT_DEPTHS = {
    "1": 0, "10": 1, "11": 1, "12": 1, "801": 2, "802": 2, "803": 2,
}
CODES = tuple(ENTITY2ID)
DOMAIN_BY_CODE = {code: "Condition" for code in CODES}
DOMAIN_BY_CODE["90"] = "Observation"  # SNOMED concept outside the domain

ZERO_UNMAPPED = {
    "missing_source": 0,
    "unsupported_source_format": 0,
    "empty_icd_mapping": 0,
    "empty_rollup": 0,
}


def make_processor(
    on_unmapped: str = "empty", max_depth: int = 1, include_root: bool = False
) -> AncestorRollupProcessor:
    """Builds the toy roll-up processor over the toy hierarchy."""
    graph = KnowledgeGraph(
        triples=TRIPLES, entity2id=ENTITY2ID, relation2id={"is_a": 0}
    )
    return AncestorRollupProcessor(
        knowledge_graph=graph,
        domain_by_code=dict(DOMAIN_BY_CODE),
        hierarchy_relation="is_a",
        root="1",
        target_policy="all_retained",
        include_root=include_root,
        max_depth=max_depth,
        on_unmapped=on_unmapped,
        root_depths=ROOT_DEPTHS,
    )


def make_mapper() -> ICDToSNOMEDMapper:
    """Builds the peer's mapper over the toy mapping table."""
    return ICDToSNOMEDMapper(
        pl.DataFrame(
            {
                "icd_version": [10, 10, 10, 10, 10, 9],
                "icd_code": ["K1", "K1", "K2", "UNKN", "OBSX", "0010"],
                "source_code": ["K1", "K1", "K2", "UNKN", "OBSX", "001.0"],
                "source_concept_id": [9001, 9001, 9002, 9005, 9006, 9003],
                "concept_id": [801, 802, 801, 999, 90, 803],
                "concept_name": [
                    "Deep X",
                    "Multi-parent C",
                    "Deep X",
                    "Outside",
                    "Observation",
                    "Deep Y",
                ],
                "target_domain_id": ["Condition"] * 6,
                "target_vocabulary_id": ["SNOMED"] * 6,
            }
        )
    )


def make_patient(patient_id: str, diagnoses) -> Patient:
    """Builds one patient from condition_occurrence row descriptions.

    Each description maps onto the event attributes the OMOP config must
    expose: ``dx`` (diagnosis id), ``concept``, ``source``, ``visit`` and
    ``ts`` (timestamp, ``TIMESTAMP`` by default).
    """
    rows = [
        {
            "dx": diagnosis["dx"],
            "concept": diagnosis.get("concept"),
            "source": diagnosis.get("source"),
            "visit": diagnosis.get("visit"),
            "ts": diagnosis.get("ts", TIMESTAMP),
        }
        for diagnosis in diagnoses
    ]
    frame = pl.DataFrame(
        {
            "patient_id": [patient_id] * len(rows),
            "event_type": ["condition_occurrence"] * len(rows),
            "timestamp": [row["ts"] for row in rows],
            "_source_row": list(range(len(rows))),
            "condition_occurrence/condition_occurrence_id": [
                row["dx"] for row in rows
            ],
            "condition_occurrence/condition_concept_id": [
                row["concept"] for row in rows
            ],
            "condition_occurrence/condition_source_value": [
                row["source"] for row in rows
            ],
            "condition_occurrence/visit_occurrence_id": [
                row["visit"] for row in rows
            ],
        },
        schema={
            "patient_id": pl.Utf8,
            "event_type": pl.Utf8,
            "timestamp": pl.Datetime,
            "_source_row": pl.Int64,
            "condition_occurrence/condition_occurrence_id": pl.Utf8,
            "condition_occurrence/condition_concept_id": pl.Utf8,
            "condition_occurrence/condition_source_value": pl.Utf8,
            "condition_occurrence/visit_occurrence_id": pl.Utf8,
        },
    )
    return Patient(patient_id=patient_id, data_source=frame)


# The exact fixture's three patients.
FIXTURE_PATIENTS = [
    make_patient(
        "1",
        [
            {"dx": "101", "source": "ICD10CM:K1", "visit": "v1"},
            {"dx": "102", "concept": "802", "visit": "v2"},
        ],
    ),
    make_patient("2", [{"dx": "201", "source": "ICD10CM:K1", "visit": "v1"}]),
    make_patient(
        "3",
        [
            {"dx": "301", "concept": "801", "visit": "v9"},
            {"dx": "302", "concept": "801", "visit": "v9"},
            {"dx": "303", "concept": "801", "visit": "v9"},
        ],
    ),
]
FIXTURE_AXES = ["11", "12", "10"]  # B, Z, A
FIXTURE_MATRIX = [[1, 0, 1], [0, 0, 0], [1, 0, 2]]


class BuilderTestCase(unittest.TestCase):
    """Shared builder arguments for the test classes below."""

    def setUp(self):
        self.processor = make_processor()
        self.mapper = make_mapper()

    def build(self, patients, axes=None, rollup=None, mapper=None):
        return build_keep_cooccurrence(
            patients,
            rollup=rollup or self.processor,
            icd_mapper=mapper or self.mapper,
            concept_ids=FIXTURE_AXES if axes is None else axes,
        )


class TestExactFixture(BuilderTestCase):
    """The handoff's three-patient fixture, axes B, Z, A."""

    def test_matrix_counts_and_layout(self):
        result = self.build(FIXTURE_PATIENTS)
        self.assertEqual(
            result.matrix.to_dense().tolist(), FIXTURE_MATRIX
        )
        self.assertEqual(result.concept_ids, ("11", "12", "10"))
        self.assertEqual(result.patients_processed, 3)
        self.assertEqual(result.patients_with_present_targets, 2)
        self.assertEqual(result.unique_diagnosis_events, 6)
        self.assertEqual(result.mapped_events, 6)
        self.assertEqual(dict(result.unmapped_events), ZERO_UNMAPPED)

    def test_matrix_layout_invariants(self):
        result = self.build(FIXTURE_PATIENTS)
        self.assertEqual(result.matrix.shape, (3, 3))
        self.assertTrue(result.matrix.is_coalesced())
        self.assertEqual(result.matrix.dtype, torch.int64)
        self.assertEqual(result.matrix.layout, torch.sparse_coo)
        self.assertEqual(result.matrix.device.type, "cpu")
        dense = result.matrix.to_dense()
        self.assertTrue(torch.equal(dense, dense.t()))
        # prevalence diagonals: A present for patients 1 and 3, B for 1
        self.assertEqual(dense[2, 2].item(), 2)
        self.assertEqual(dense[0, 0].item(), 1)
        self.assertEqual(dense[1, 1].item(), 0)

    def test_single_pass_generator_input(self):
        result = self.build(iter(FIXTURE_PATIENTS))
        self.assertEqual(
            result.matrix.to_dense().tolist(), FIXTURE_MATRIX
        )
        self.assertEqual(result.patients_processed, 3)


class TestDiagnosisIdentity(BuilderTestCase):
    """Diagnosis-row identity: collapse, conflict, and visit rules."""

    def test_identical_copies_collapse(self):
        patient = make_patient(
            "1",
            [
                {"dx": "101", "concept": "801", "visit": "v1"},
                {"dx": "101", "concept": "801", "visit": "v1"},
                {"dx": "102", "concept": "801", "visit": "v1"},
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        self.assertEqual(result.unique_diagnosis_events, 2)
        self.assertEqual(result.mapped_events, 2)
        # two distinct diagnoses (101, 102) establish A exactly once
        self.assertEqual(result.matrix.to_dense()[0, 0].item(), 1)

    def test_conflicting_copies_are_rejected(self):
        base = {"dx": "101", "concept": "801", "visit": "v1"}
        variants = [
            {"concept": "803"},
            {"source": "ICD10CM:K2"},
            {"ts": datetime(2021, 5, 5, 12, 0, 0)},
            {"visit": "v2"},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                patient = make_patient("1", [base, {**base, **variant}])
                with self.assertRaises(ValueError):
                    self.build([patient])

    def test_missing_diagnosis_id_is_rejected(self):
        for missing in (None, "", "None"):
            with self.subTest(value=missing):
                patient = make_patient(
                    "1", [{"dx": missing, "concept": "801", "visit": "v1"}]
                )
                with self.assertRaises(ValueError):
                    self.build([patient])

    def test_events_without_visit_id_are_included(self):
        patient = make_patient(
            "1",
            [
                {"dx": "101", "concept": "801"},
                {"dx": "102", "concept": "801"},
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        self.assertEqual(result.unique_diagnosis_events, 2)
        self.assertEqual(result.matrix.to_dense()[0, 0].item(), 1)

    def test_distinct_rows_same_visit_and_code_are_preserved(self):
        patient = make_patient(
            "1",
            [
                {"dx": "101", "concept": "801", "visit": "v1"},
                {"dx": "102", "concept": "801", "visit": "v1"},
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        self.assertEqual(result.unique_diagnosis_events, 2)
        self.assertEqual(result.matrix.to_dense()[0, 0].item(), 1)

    def test_diagnosis_ids_are_scoped_to_a_patient(self):
        # the peer's row numbering is per table, but the builder treats
        # diagnosis ids as patient-scoped: the same id in another patient
        # is a different diagnosis, not a conflict
        first = make_patient(
            "1", [{"dx": "101", "concept": "801", "visit": "v1"}]
        )
        second = make_patient(
            "2", [{"dx": "101", "concept": "801", "visit": "v1"}]
        )
        result = self.build([first, second], axes=["10", "11", "12"])
        self.assertEqual(result.patients_processed, 2)
        self.assertEqual(result.unique_diagnosis_events, 2)
        # one diagnosis each: nothing present
        self.assertEqual(result.matrix.to_dense().sum().item(), 0)

    def test_repeated_patients_are_rejected(self):
        patient = FIXTURE_PATIENTS[0]
        with self.assertRaises(ValueError):
            self.build([patient, patient])

    def test_patient_without_condition_events_is_counted(self):
        frame = pl.DataFrame(
            {
                "patient_id": ["9"],
                "event_type": ["visit_occurrence"],
                "timestamp": [TIMESTAMP],
                "_source_row": [0],
                "visit_occurrence/visit_occurrence_id": ["v1"],
            },
            schema={
                "patient_id": pl.Utf8,
                "event_type": pl.Utf8,
                "timestamp": pl.Datetime,
                "_source_row": pl.Int64,
                "visit_occurrence/visit_occurrence_id": pl.Utf8,
            },
        )
        result = self.build([Patient(patient_id="9", data_source=frame)])
        self.assertEqual(result.patients_processed, 1)
        self.assertEqual(result.unique_diagnosis_events, 0)
        self.assertEqual(result.mapped_events, 0)
        self.assertEqual(result.matrix.to_dense().sum().item(), 0)


class TestEventMapping(BuilderTestCase):
    """Standard-concept precedence, ICD fallback, and unmapped reasons."""

    def test_standard_concept_id_takes_precedence(self):
        # concept 803 rolls up to {"11"}; the source value would map to
        # 801 -> {"10"}; the canonical concept must win.
        patient = make_patient(
            "1",
            [
                {
                    "dx": "101",
                    "concept": "803",
                    "source": "ICD10CM:I25X",
                    "visit": "v1",
                },
                {
                    "dx": "102",
                    "concept": "803",
                    "source": "ICD10CM:I25X",
                    "visit": "v1",
                },
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        self.assertEqual(
            result.matrix.to_dense().tolist(),
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        )

    def test_icd9_and_icd10_versions_with_leading_zeros(self):
        patient = make_patient(
            "1",
            [
                {"dx": "101", "source": "ICD9CM:0010", "visit": "v1"},
                {"dx": "102", "source": "ICD10CM:k2", "visit": "v1"},
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        # both codes map (the mapper normalizes the lowercase ICD-10 code);
        # one diagnosis each establishes no presence
        self.assertEqual(result.mapped_events, 2)
        self.assertEqual(result.matrix.to_dense().sum().item(), 0)
        # the leading-zero ICD-9 code reached 803 -> {"11"}: paired with a
        # canonical 803 diagnosis it establishes 11
        patient2 = make_patient(
            "1",
            [
                {"dx": "103", "source": "ICD9CM:0010", "visit": "v1"},
                {"dx": "104", "concept": "803", "visit": "v2"},
            ],
        )
        result2 = self.build([patient2], axes=["10", "11", "12"])
        self.assertEqual(
            result2.matrix.to_dense().tolist(),
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        )

    def test_each_unmapped_reason(self):
        patient = make_patient(
            "4",
            [
                {"dx": "401"},
                {"dx": "402", "source": "SNOMED:123"},
                {"dx": "403", "source": "ICD10CM:NOPE"},
                {"dx": "404", "source": "ICD10CM:UNKN"},
                # OBSX maps to the Observation concept 90: known code,
                # outside the eligible domain, so the roll-up is empty
                {"dx": "405", "source": "ICD10CM:OBSX"},
            ],
        )
        result = self.build([patient])
        self.assertEqual(
            dict(result.unmapped_events),
            {
                "missing_source": 1,
                "unsupported_source_format": 1,
                "empty_icd_mapping": 1,
                "empty_rollup": 2,
            },
        )
        self.assertEqual(result.mapped_events, 0)
        self.assertEqual(result.unique_diagnosis_events, 5)
        self.assertEqual(
            result.mapped_events + sum(result.unmapped_events.values()),
            result.unique_diagnosis_events,
        )
        self.assertEqual(result.patients_with_present_targets, 0)
        self.assertEqual(result.matrix.to_dense().sum().item(), 0)

    def test_strict_rollup_errors_propagate(self):
        strict = make_processor(on_unmapped="error")
        canonical = make_patient(
            "1", [{"dx": "101", "concept": "999", "visit": "v1"}]
        )
        with self.assertRaises(UnmappedCodeError):
            self.build([canonical], rollup=strict)
        icd = make_patient("1", [{"dx": "102", "source": "ICD10CM:UNKN"}])
        with self.assertRaises(UnmappedCodeError):
            self.build([icd], rollup=strict)

    def test_union_across_icd_targets_counts_each_target_once(self):
        # K1 maps to 801 and 802; both routes reach 10, 802 also reaches 11.
        # One K1 diagnosis plus one canonical 801 diagnosis: 10 is present
        # through two distinct diagnoses, 11 is not, and the converging
        # routes do not double-count 10.
        patient = make_patient(
            "1",
            [
                {"dx": "101", "source": "ICD10CM:K1", "visit": "v1"},
                {"dx": "102", "concept": "801", "visit": "v1"},
            ],
        )
        result = self.build([patient], axes=["10", "11", "12"])
        self.assertEqual(
            result.matrix.to_dense().tolist(),
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        )


class TestAxes(BuilderTestCase):
    """Axis validation, ordering, and empty-input shapes."""

    def test_reordered_axes_permute_the_matrix(self):
        result = self.build(FIXTURE_PATIENTS, axes=["10", "12", "11"])
        self.assertEqual(result.concept_ids, ("10", "12", "11"))
        self.assertEqual(
            result.matrix.to_dense().tolist(),
            [[2, 0, 1], [0, 0, 0], [1, 0, 1]],
        )

    def test_invalid_axes_are_rejected(self):
        patient = FIXTURE_PATIENTS[0]
        cases = [
            ({"10", "11", "12"}, TypeError),
            ("10", TypeError),
            (iter(["10", "11", "12"]), TypeError),
            (["10", 11, "12"], TypeError),
            (["10", "", "12"], ValueError),
            (["10", "11"], ValueError),
            (["10", "11", "12", "10"], ValueError),
            (["10", "11", "12", "999"], ValueError),
        ]
        for axes, error in cases:
            with self.subTest(axes=axes), self.assertRaises(error):
                self.build([patient], axes=axes)

    def test_empty_cohort_keeps_vocabulary_shape(self):
        result = self.build([])
        self.assertEqual(result.matrix.shape, (3, 3))
        self.assertEqual(result.matrix.to_dense().sum().item(), 0)
        self.assertEqual(result.patients_processed, 0)
        self.assertEqual(result.patients_with_present_targets, 0)
        self.assertEqual(result.unique_diagnosis_events, 0)
        self.assertEqual(result.mapped_events, 0)
        self.assertEqual(dict(result.unmapped_events), ZERO_UNMAPPED)
        self.assertTrue(result.matrix.is_coalesced())

    def test_empty_vocabulary_yields_zero_by_zero(self):
        empty = make_processor(max_depth=0, include_root=False)
        result = build_keep_cooccurrence(
            [], rollup=empty, icd_mapper=self.mapper, concept_ids=[]
        )
        self.assertEqual(result.matrix.shape, (0, 0))
        self.assertEqual(result.concept_ids, ())
        self.assertTrue(result.matrix.is_coalesced())


class TestRetainedCodesProperty(unittest.TestCase):
    """The processor's retained vocabulary, by cutoff and root inclusion."""

    def test_depth_cutoff_and_root_inclusion(self):
        shallow = make_processor(max_depth=1, include_root=False)
        self.assertEqual(shallow.retained_codes, frozenset({"10", "11", "12"}))
        deep = make_processor(max_depth=2, include_root=False)
        self.assertEqual(
            deep.retained_codes,
            frozenset({"10", "11", "12", "801", "802", "803"}),
        )
        with_root = make_processor(max_depth=1, include_root=True)
        self.assertEqual(
            with_root.retained_codes, frozenset({"1", "10", "11", "12"})
        )

    def test_read_only(self):
        processor = make_processor()
        self.assertIsInstance(processor.retained_codes, frozenset)
        with self.assertRaises(AttributeError):
            processor.retained_codes = {"999"}
