# Author: lee-leonardo (NetID: ll37)

"""Build a KEEP matrix from a synthetic MIMIC-III patient cohort.

Run from the checkout with ``python -m examples.mimic3_keep_cooccurrence_matrix``.
The histories show how two diagnosis rows with the same ICD-9 code stay
distinct by ``row_id``, while an exact copy collapses.
"""

from examples._keep_mimic_example import CONFIGS, run_example
from pyhealth.datasets import MIMIC3Dataset


def load_dataset(root, cache_dir):
    return MIMIC3Dataset(
        root=str(root),
        tables=["diagnoses_icd"],
        config_path=str(CONFIGS / "mimic3_keep.yaml"),
        cache_dir=str(cache_dir),
    )


def main():
    return run_example(3, load_dataset)


if __name__ == "__main__":
    main()
