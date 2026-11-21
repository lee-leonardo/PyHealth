# Author: lee-leonardo (NetID: ll37)

"""Build a KEEP matrix from a synthetic MIMIC-IV patient cohort.

Run from the checkout with ``python -m examples.mimic4_keep_cooccurrence_matrix``.
The histories show how diagnoses that share an admission and sequence
stay distinct when their ICD codes differ. An exact copy collapses.
"""

from examples._keep_mimic_example import CONFIGS, run_example
from pyhealth.datasets import MIMIC4Dataset


def load_dataset(root, cache_dir):
    return MIMIC4Dataset(
        ehr_root=str(root),
        ehr_tables=["diagnoses_icd"],
        ehr_config_path=str(CONFIGS / "mimic4_keep.yaml"),
        cache_dir=str(cache_dir),
    )


def main():
    return run_example(4, load_dataset)


if __name__ == "__main__":
    main()
