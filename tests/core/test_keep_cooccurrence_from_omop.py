"""Integrated KEEP co-occurrence tests on the real OMOP readers.

The Athena bundle and the OMOP config come from the documented example
(``examples/omop_keep_cooccurrence_matrix.py``), which owns the synthetic
vocabulary: the Disease root 4274025, a depth-6 deep source (``110``), a
multi-parent concept (``104``), an isolated retained orphan (``105``), and
one standard-stitched closure edge that only ``path_source="closure"``
can traverse (``250``). This file adds its own condition cohort and
asserts the complete matrix, the axis order, the aggregate counts, the
processor's ``retained_codes`` property through the depth cutoff and both
root-inclusion settings, and the documented example's output.

Cohort expectations: persons 1 and 3 make the whole deep chain
``{100, 101, 102, 108, 109}`` present (110's retained ancestors under
``all_retained``), person 2's single converging diagnosis establishes
nothing, and person 4 exercises every unmapped reason plus two no-visit
diagnoses of the retained multi-parent concept ``104``.
"""

import importlib.util
import shutil
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import torch

from pyhealth.datasets import OMOPDataset
from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import build_keep_cooccurrence, build_keep_omop_ancestor_rollup

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "omop_keep_cooccurrence_matrix.py"
)
_spec = importlib.util.spec_from_file_location(
    "omop_keep_cooccurrence_matrix", EXAMPLE_PATH
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

ROOT = example.ROOT

# condition_occurrence_id, person_id, condition_concept_id (0/empty = ICD
# source value), condition_start_datetime, visit_occurrence_id (empty =
# none), condition_source_value
CONDITIONS = [
    (201, 1, "110", "2118-01-05 09:30:00", "11", ""),
    (202, 1, "0", "2118-01-06 10:30:00", "11", "ICD10CM:K1"),
    (203, 1, "0", "2118-06-10 11:00:00", "12", "ICD9CM:0010"),
    (204, 1, "105", "2118-06-10 11:30:00", "12", ""),
    (301, 2, "0", "2118-02-01 08:30:00", "21", "ICD10CM:K1"),
    (401, 3, "110", "2118-03-11 07:30:00", "31", ""),
    (402, 3, "110", "2118-03-12 07:30:00", "31", ""),
    (403, 3, "110", "2118-03-13 07:30:00", "31", ""),
    (501, 4, "", "2118-04-20 06:30:00", "41", ""),
    (502, 4, "0", "2118-04-20 06:31:00", "41", "SNOMED:4274025"),
    (503, 4, "0", "2118-04-20 06:32:00", "41", "ICD10CM:Z999"),
    (505, 4, "4105351", "2118-04-20 06:33:00", "41", ""),
    (506, 4, "104", "2118-04-20 06:34:00", "", ""),
    (507, 4, "104", "2118-04-20 06:35:00", "", ""),
]

# visit_occurrence_id, person_id, start, end
VISITS = [
    (11, 1, "2118-01-05 09:00:00", "2118-01-08 09:00:00"),
    (12, 1, "2118-06-10 10:00:00", "2118-06-11 10:00:00"),
    (21, 2, "2118-02-01 08:00:00", "2118-02-02 08:00:00"),
    (31, 3, "2118-03-11 07:00:00", "2118-03-15 07:00:00"),
    (41, 4, "2118-04-20 06:00:00", "2118-04-21 06:00:00"),
]

# The concepts persons 1 and 3 establish by rolling 110 up: all retained
# ancestors of the depth-6 source under all_retained.
DEEP_CHAIN = ("100", "101", "102", "108", "109")


def _pair_counts(codes, count):
    """Maps every unordered pair of ``codes`` to ``count``."""
    return {pair: count for pair in combinations(sorted(codes), 2)}


def expected_dense(concept_ids, diagonal, pairs):
    """Builds the expected dense matrix from diagonal and pair counts.

    ``diagonal`` maps concept id to prevalence; ``pairs`` maps an unordered
    ``(left, right)`` pair to its patient count, mirrored on both
    off-diagonal entries.
    """
    dense = [[0] * len(concept_ids) for _ in concept_ids]
    index = {code: position for position, code in enumerate(concept_ids)}
    for code, count in diagonal.items():
        dense[index[code]][index[code]] = count
    for (left, right), count in pairs.items():
        dense[index[left]][index[right]] = count
        dense[index[right]][index[left]] = count
    return dense


class TestFromOmopCooccurrence(unittest.TestCase):
    """The complete OMOP -> ICD -> roll-up -> matrix path on real readers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="keep_cooccurrence_omop_"))
        cls.athena = cls.tmp / "athena"
        example.write_athena_bundle(cls.athena)
        cls.vocabulary = OMOPVocabulary(cls.athena)
        cls.processor = build_keep_omop_ancestor_rollup(
            cls.vocabulary,
            target_policy="all_retained",
            include_root=False,
            max_depth=5,
            path_source="closure",
            on_unmapped="empty",
        )
        cls.mapper = cls.vocabulary.icd_mapper()
        cls.omop = cls.tmp / "omop"
        config_path = cls.write_cohort(cls.omop)
        cls.dataset = OMOPDataset(
            root=str(cls.omop),
            tables=["condition_occurrence"],
            config_path=str(config_path),
            cache_dir=str(cls.tmp / "cache"),
        )
        cls.concept_ids = sorted(cls.processor.retained_codes)
        cls.result = build_keep_cooccurrence(
            cls.dataset.iter_patients(),
            rollup=cls.processor,
            icd_mapper=cls.mapper,
            concept_ids=cls.concept_ids,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def write_cohort(cls, folder: Path) -> Path:
        """Writes this test's OMOP CSVs; returns the config path."""
        folder.mkdir(parents=True, exist_ok=True)
        example._write(
            folder / "person.csv",
            ["person_id", "year_of_birth", "month_of_birth", "day_of_birth"],
            [(1, 1950, 1, 1), (2, 1960, 1, 1), (3, 1970, 1, 1), (4, 1980, 1, 1)],
            separator=",",
        )
        example._write(
            folder / "visit_occurrence.csv",
            [
                "visit_occurrence_id", "person_id", "visit_start_datetime",
                "visit_end_datetime",
            ],
            VISITS,
            separator=",",
        )
        example._write(
            folder / "death.csv",
            ["person_id", "death_datetime"],
            [],
            separator=",",
        )
        example._write(
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
        config_path.write_text(example.OMOP_CONFIG, encoding="utf-8")
        return config_path

    def test_retained_vocabulary(self):
        # depth <= 5 from the Disease root, minimum closure measure, root
        # excluded: 12 concepts
        self.assertEqual(
            self.processor.retained_codes,
            frozenset(
                {
                    "100", "111", "101", "104", "222", "102", "105", "333",
                    "108", "444", "109", "140708",
                }
            ),
        )

    def test_retained_codes_reflect_cutoff_and_root_inclusion(self):
        shallow = build_keep_omop_ancestor_rollup(
            self.vocabulary,
            target_policy="all_retained",
            include_root=False,
            max_depth=3,
            path_source="closure",
            on_unmapped="empty",
        )
        self.assertEqual(
            shallow.retained_codes,
            frozenset(
                {"100", "111", "101", "104", "222", "102", "105", "333"}
            ),
        )
        with_root = build_keep_omop_ancestor_rollup(
            self.vocabulary,
            target_policy="all_retained",
            include_root=True,
            max_depth=1,
            path_source="closure",
            on_unmapped="empty",
        )
        self.assertEqual(
            with_root.retained_codes, frozenset({str(ROOT), "100", "111"})
        )

    def test_axis_order(self):
        self.assertEqual(self.result.concept_ids, tuple(self.concept_ids))
        self.assertEqual(len(self.result.concept_ids), 12)

    def test_complete_matrix(self):
        expected = expected_dense(
            self.concept_ids,
            # prevalence: the deep chain for persons 1 and 3, plus 104 for
            # person 4's two no-visit diagnoses; 105 never reaches two
            # distinct diagnoses
            diagonal={
                **{code: 2 for code in DEEP_CHAIN},
                "104": 1,
            },
            pairs=_pair_counts(DEEP_CHAIN, 2),
        )
        self.assertEqual(self.result.matrix.to_dense().tolist(), expected)

    def test_counts(self):
        self.assertEqual(self.result.patients_processed, 4)
        self.assertEqual(self.result.patients_with_present_targets, 3)
        self.assertEqual(self.result.unique_diagnosis_events, 14)
        self.assertEqual(self.result.mapped_events, 10)
        self.assertEqual(
            dict(self.result.unmapped_events),
            {
                "missing_source": 1,
                "unsupported_source_format": 1,
                "empty_icd_mapping": 1,
                "empty_rollup": 1,
            },
        )

    def test_matrix_layout_invariants(self):
        matrix = self.result.matrix
        self.assertEqual(matrix.shape, (12, 12))
        self.assertTrue(matrix.is_coalesced())
        self.assertEqual(matrix.dtype, torch.int64)
        self.assertEqual(matrix.device.type, "cpu")
        dense = matrix.to_dense()
        self.assertTrue(torch.equal(dense, dense.t()))

    def test_documented_example(self):
        workdir = self.tmp / "example"
        result = example.main(["--workdir", str(workdir)])
        self.assertEqual(result.matrix.shape, (12, 12))
        self.assertEqual(result.concept_ids, tuple(self.concept_ids))
        self.assertEqual(result.patients_processed, 5)
        self.assertEqual(result.patients_with_present_targets, 3)
        self.assertEqual(result.unique_diagnosis_events, 11)
        self.assertEqual(result.mapped_events, 9)
        self.assertEqual(
            dict(result.unmapped_events),
            {
                "missing_source": 1,
                "unsupported_source_format": 0,
                "empty_icd_mapping": 1,
                "empty_rollup": 0,
            },
        )
        # the deep chain for persons 1 and 3; the stitched closure path
        # (ICD I250 -> 250 -> retained ancestors) for person 4
        stitched = ("111", "140708", "222", "333", "444")
        expected = expected_dense(
            result.concept_ids,
            diagonal={
                **{code: 2 for code in DEEP_CHAIN},
                **{code: 1 for code in stitched},
            },
            pairs={**_pair_counts(DEEP_CHAIN, 2), **_pair_counts(stitched, 1)},
        )
        self.assertEqual(result.matrix.to_dense().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
