# Author: lee-leonardo (NetID: ll37)

"""Shared synthetic data and workflow for the two MIMIC KEEP examples."""

import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.omop_keep_cooccurrence_matrix import report, write_athena_bundle
from pyhealth import datasets
from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import KeepConditionHistoryTask, build_keep_omop_ancestor_rollup

CONFIGS = Path(datasets.__file__).with_name("configs")


def _write_table(root, filename, fields, rows):
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _write_mimic_cohort(root, version):
    patients = [("1",), ("2",)]
    admitted = "2020-01-01 09:00:00"
    admissions = [("1", "11", admitted), ("2", "21", admitted)]
    stays = [("1", admitted)]

    if version == 3:
        _write_table(root, "PATIENTS.csv.gz", ("SUBJECT_ID",), patients)
        _write_table(
            root,
            "ADMISSIONS.csv.gz",
            ("SUBJECT_ID", "HADM_ID", "ADMITTIME"),
            admissions,
        )
        _write_table(root, "ICUSTAYS.csv.gz", ("SUBJECT_ID", "INTIME"), stays)
        _write_table(
            root,
            "DIAGNOSES_ICD.csv.gz",
            ("SUBJECT_ID", "HADM_ID", "ROW_ID", "ICD9_CODE"),
            [
                ("1", "11", "101", "0010"),
                ("1", "11", "101", "0010"),  # Exact copy.
                ("1", "11", "102", "0010"),  # Same code, distinct diagnosis.
                ("2", "21", "201", "0010"),
            ],
        )
    else:
        _write_table(root, "hosp/patients.csv.gz", ("subject_id",), patients)
        _write_table(
            root,
            "hosp/admissions.csv.gz",
            ("subject_id", "hadm_id", "admittime"),
            admissions,
        )
        _write_table(root, "icu/icustays.csv.gz", ("subject_id", "intime"), stays)
        _write_table(
            root,
            "hosp/diagnoses_icd.csv.gz",
            ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
            [
                ("1", "11", "1", "K1", "10"),
                ("1", "11", "1", "K1", "10"),  # Exact copy.
                ("1", "11", "1", "I214", "10"),  # Same sequence, distinct diagnosis.
                ("2", "21", "1", "K1", "10"),  # One row maps to multiple concepts.
            ],
        )


def _report_histories(histories):
    print("Mapped patient histories (synthetic)")
    for history in histories:
        print(f"Patient {history['patient_id']}")
        for diagnosis in history["conditions"]:
            print(
                f"  {diagnosis['diagnosis_id']!r} in visit {diagnosis['visit_id']} "
                f"at {diagnosis['timestamp']}: "
                f"{diagnosis['source_value']} -> {diagnosis['concept_ids']}"
            )
    print()


def run_example(version, load_dataset):
    """Build, display, and reuse the synthetic histories for KEEP counting."""
    with TemporaryDirectory(prefix=f"keep_mimic{version}_") as temporary:
        workdir = Path(temporary)
        root = workdir / "mimic"
        _write_mimic_cohort(root, version)
        athena = workdir / "athena"
        write_athena_bundle(athena)
        vocabulary = OMOPVocabulary(athena)
        rollup = build_keep_omop_ancestor_rollup(
            vocabulary,
            target_policy="all_retained",
            include_root=False,
            max_depth=2,  # Keep the displayed matrix small.
            path_source="closure",
            on_unmapped="empty",
        )
        dataset = load_dataset(root, workdir / "cache")
        task = KeepConditionHistoryTask(
            mimic_version=version,
            vocabulary=vocabulary,
            vocabulary_release="synthetic",
        )
        histories = dataset.set_task(task)
        _report_histories(histories)
        result = task.build_cooccurrence(
            histories,
            rollup=rollup,
            concept_ids=sorted(rollup.retained_codes),
        )
        report(result)
        return result
