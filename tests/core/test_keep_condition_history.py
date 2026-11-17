"""Synthetic MIMIC reader, history cache, and KEEP counting tests."""

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pyhealth import datasets
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.graph import KnowledgeGraph
from pyhealth.medcode import OMOPVocabulary
from pyhealth.processors import AncestorRollupProcessor
from pyhealth.tasks import KeepConditionHistoryTask
from tests.core.test_keep_cooccurrence_from_omop import example

CONFIGS = Path(datasets.__file__).with_name("configs")
ADMITTED = datetime(2020, 1, 1, 9)
AXES = ("B", "Z", "A")


def _write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def vocabulary(tmp_path):
    athena = tmp_path / "athena"
    example.write_athena_bundle(athena)
    # Match the existing ICD-9 source text and target in ICD-10 as well.
    with (athena / "CONCEPT.csv").open("a", encoding="utf-8") as stream:
        stream.write("9007\tFixture\tCondition\tICD10CM\tClinical Finding\t\t0010\t\n")
    with (athena / "CONCEPT_RELATIONSHIP.csv").open("a", encoding="utf-8") as stream:
        stream.write("9007\t109\tMaps to\t\n")
    return OMOPVocabulary(athena)


class _LegacyCacheTask(KeepConditionHistoryTask):
    """Seed a v1 cache with distinguishable synthetic histories."""

    task_name = "keep_condition_history_v1"

    def __call__(self, patient):
        return [{"patient_id": patient.patient_id, "conditions": []}]


def _by_patient(histories):
    return {
        sample["patient_id"]: sorted(
            sample["conditions"], key=lambda record: record["diagnosis_id"]
        )
        for sample in histories
    }


def test_mimic4_full_key_and_versioned_history_cache(tmp_path, vocabulary):
    root = tmp_path / "mimic4"
    patients = ("1", "2", "3")
    _write_csv(root / "hosp/patients.csv.gz", ("subject_id",), [(p,) for p in patients])
    _write_csv(
        root / "hosp/admissions.csv.gz",
        ("subject_id", "hadm_id", "admittime"),
        [(p, p + "1", ADMITTED) for p in patients],
    )
    _write_csv(
        root / "icu/icustays.csv.gz", ("subject_id", "intime"), [("1", ADMITTED)]
    )
    _write_csv(
        root / "hosp/diagnoses_icd.csv.gz",
        ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
        [
            ("1", "11", "1", "0010", "9"),
            ("1", "11", "1", "0010", "10"),
            ("1", "11", "1", "0010", "9"),
            ("2", "21", "1", "k.1", "10"),
            ("2", "21", "1", "I214", "10"),
            ("3", "31", "1", "k.1", "10"),
            ("3", "31", "1", "k.1", "10"),
        ],
    )
    dataset = MIMIC4Dataset(
        ehr_root=str(root),
        ehr_tables=["diagnoses_icd"],
        ehr_config_path=str(CONFIGS / "mimic4_keep.yaml"),
        cache_dir=str(tmp_path / "cache"),
        num_workers=1,
    )
    task_args = {
        "mimic_version": 4,
        "vocabulary": vocabulary,
        "vocabulary_release": "fixture",
    }
    legacy = _LegacyCacheTask(**task_args)
    assert _by_patient(dataset.set_task(legacy)) == {p: [] for p in patients}
    cache_dir = dataset.cache_dir / "tasks"
    (legacy_cache,) = cache_dir.glob("keep_condition_history_v1_*")
    assert (legacy_cache / "task_df.ld" / "index.json").is_file()

    task = KeepConditionHistoryTask(**task_args)
    records = _by_patient(dataset.set_task(task))
    assert {p: len(rows) for p, rows in records.items()} == {"1": 2, "2": 2, "3": 1}
    assert {
        p: {tuple(json.loads(row["diagnosis_id"])) for row in rows}
        for p, rows in records.items()
    } == {
        "1": {("11", "1", "0010", "9"), ("11", "1", "0010", "10")},
        "2": {("21", "1", "k.1", "10"), ("21", "1", "I214", "10")},
        "3": {("31", "1", "k.1", "10")},
    }
    assert {row["source_value"] for row in records["1"]} == {
        "ICD9CM:0010",
        "ICD10CM:0010",
    }
    assert all(
        row["visit_id"] == p + "1" and row["timestamp"] == ADMITTED
        for p, rows in records.items()
        for row in rows
    )
    assert all(row["concept_ids"] == ["109"] for row in records["1"])
    assert records["3"][0]["concept_ids"] == ["104", "110"]
    (current_cache,) = cache_dir.glob("keep_condition_history_v2_*")
    assert (current_cache / "task_df.ld" / "index.json").is_file()
    assert (legacy_cache / "task_df.ld" / "index.json").is_file()

    cached = dataset.set_task(KeepConditionHistoryTask(**task_args))
    assert _by_patient(cached) == records
    triples = [("R", "is_a", code) for code in AXES] + [
        (parent, "is_a", code)
        for parent in ("A", "B")
        for code in ("104", "109", "110")
    ]
    rollup = AncestorRollupProcessor(
        knowledge_graph=KnowledgeGraph(triples=triples),
        domain_by_code=dict.fromkeys(("R", *AXES, "104", "109", "110"), "Condition"),
        hierarchy_relation="is_a",
        root="R",
        target_policy="all_retained",
        include_root=False,
        max_depth=1,
    )
    result = task.build_cooccurrence(cached, rollup=rollup, concept_ids=AXES)
    assert result.concept_ids == AXES
    assert result.matrix.layout == torch.sparse_coo
    assert result.matrix.is_coalesced()
    assert result.matrix.to_dense().tolist() == [[2, 0, 2], [0, 0, 0], [2, 0, 2]]
    assert result.patients_processed == 3
    assert result.patients_with_present_targets == 2
    assert result.unique_diagnosis_events == result.mapped_events == 5
    assert not any(result.unmapped_events.values())


def test_mimic4_preserves_missing_code_or_version(vocabulary):
    task = KeepConditionHistoryTask(
        mimic_version=4, vocabulary=vocabulary, vocabulary_release="fixture"
    )
    events = [
        SimpleNamespace(hadm_id="11", seq_num="1", icd_code=None, icd_version="9"),
        SimpleNamespace(hadm_id="11", seq_num="1", icd_code="0010", icd_version=None),
    ]
    patient = SimpleNamespace(
        patient_id="1",
        get_events=lambda event_type: events if event_type == "diagnoses_icd" else [],
    )
    conditions = task(patient)[0]["conditions"]
    assert {
        tuple(json.loads(row["diagnosis_id"])): row["unmapped_reason"]
        for row in conditions
    } == {
        ("11", "1", None, "9"): "missing_source",
        ("11", "1", "0010", None): "unsupported_source_format",
    }


def test_mimic3_keeps_row_ids(tmp_path, vocabulary):
    root = tmp_path / "mimic3"
    _write_csv(root / "PATIENTS.csv.gz", ("SUBJECT_ID",), [("1",)])
    _write_csv(
        root / "ADMISSIONS.csv.gz",
        ("SUBJECT_ID", "HADM_ID", "ADMITTIME"),
        [("1", "11", ADMITTED)],
    )
    _write_csv(root / "ICUSTAYS.csv.gz", ("SUBJECT_ID", "INTIME"), [("1", ADMITTED)])
    _write_csv(
        root / "DIAGNOSES_ICD.csv.gz",
        ("SUBJECT_ID", "HADM_ID", "ROW_ID", "ICD9_CODE"),
        [
            ("1", "11", "101", "0010"),
            ("1", "11", "103", "0010"),
            ("1", "11", "101", "0010"),
        ],
    )
    dataset = MIMIC3Dataset(
        root=str(root),
        tables=["diagnoses_icd"],
        config_path=str(CONFIGS / "mimic3_keep.yaml"),
        cache_dir=str(tmp_path / "cache"),
        num_workers=1,
    )
    task = KeepConditionHistoryTask(
        mimic_version=3, vocabulary=vocabulary, vocabulary_release="fixture"
    )
    records = _by_patient(dataset.set_task(task))["1"]
    assert [row["diagnosis_id"] for row in records] == ["101", "103"]
    assert all(row["concept_ids"] == ["109"] for row in records)
