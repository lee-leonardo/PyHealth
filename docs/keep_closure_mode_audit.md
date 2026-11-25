# Graph closure analysis: graph construction and mapping coverage

Measured 2026-11-25. **Recommendation: retain the closure option on the evidence
from this bundle; do not change the default.** It supplies 334 additional edges,
recovers 26 deep sources that otherwise have no target, and changes additional
nonempty target sets. This establishes a vocabulary-level benefit in mapping
coverage and a behavioral distinction. It does not establish clinical benefit,
diagnosis-weighted impact, embedding quality, or author-artifact fidelity.

## Contribution framing and next methods

The proposed PyHealth contribution is a reproducible validation workflow for
how ontology traversal configuration affects generated graph structure and
mapping coverage. The present paired comparison is the initial evidence;
learning implications are discussed in the contribution plan. The
complete population-wide projected-graph comparison and the methodological
extensions in [the contribution plan](graph_closure_contribution_plan.md) remain
unimplemented. No model-training or predictive-performance claim is needed to
establish the workflow's utility.

Use **mapping failure rate**, rather than semantic error rate, for `no_target`
counts. On the measured deep population the rates are 33/23,230 = 0.142% and
7/23,230 = 0.030%, an absolute reduction of 0.112 percentage points and a
relative reduction of 78.8%. A newly reachable target may improve coverage;
its clinical correctness is not established by reachability alone. Distinguish
these failures from expected root exclusion, external-code classification,
invalid graph construction and projection-invariant violations.

The next contribution milestone pairs exact generated-graph comparisons with
an independent synthetic reference and explicit denominator reporting. Depth
sensitivity, controlled structural perturbations and release replication are
specified separately so their proposed results cannot be confused with
measurements already completed.

## What is being compared

`U` is the uncut Disease subtree of valid standard SNOMED Condition concepts,
rooted at OMOP concept 4274025. `D` contains valid direct `Is a` edges between
these nodes, oriented parent → child. `P` contains unique `CONCEPT_ANCESTOR`
pairs with minimum separation **exactly one**, with both endpoints in `U`.

Direct traversal selects `D`; closure traversal selects `D ∪ P`. Direct
provenance wins on overlap. This option does not take the arbitrary transitive
closure and does not enable orphan rescue. The generic processor consumes the
graph and supplied pairs; the KEEP factory owns OMOP extraction and filtering.

Both modes use the same authoritative minimum root separations, target cutoff
five, excluded root, and disabled orphan rescue. Deep sources and intermediate
nodes remain traversable. `first_retained` stops at the first eligible node on
each branch; `all_retained` continues through eligible nodes. Neither policy is
author-verified by this experiment. Every source in `U` was evaluated once per
mode and policy, in sorted string-ID order, in batches of 512.

Diagnostic enumeration uses `on_unmapped="empty"`; production defaults to
`"error"`. A synthetic check verifies that the actual default raises on a
failure. No failures are silently discarded: aggregate counts retain the full
population denominator, and local failure tables retain every failed source.

## Input and implementation identity

The user explicitly supplied a local Athena export. These are full-table
observations on that bundle, with no sampling. “Full” refers to its entire
Disease subtree, not to every vocabulary available from Athena.

- Tested production revision: `33c2cb635572a493dae428c4666a5ec503c5ee77`.
- Supplied worktree: clean, detached HEAD before this work; audit files added
  locally for measurement, with no production edits. The measured run preceded
  publication of the helper; the subsequently scrubbed implementation is commit
  `b2a5150` on `task/closure_mode_investigation`. Documentation is now prepared for circulation.
- This is the earlier unoptimized factory, not the loader optimization's `3c86e39` factory.
  Performance numbers must not be attributed to that optimization.
- `VOCABULARY.csv` identifies OMOP Standardized Vocabularies as
  `v5.0 29-AUG-26` (the `None` vocabulary row). Its `Vocabulary` row has no
  version value; the file modification/download date is not a release identity.
- SNOMED version: `2026-02-01 SNOMED CT International Edition; 2026-03-01
  SNOMED CT US Edition; 2026-04-08 SNOMED CT UK Edition`.
- Python 3.12.12; PyHealth 2.1.0 from this checkout; Polars 1.35.2;
  PyTorch 2.15.0; NetworkX 3.7; torch-geometric 2.8.1.
- macOS 27.2.1, arm64, 18 logical CPUs reported by Python. No device model or
  physical RAM measurement was obtained.

| Input table | Bytes | SHA-256 |
| --- | ---: | --- |
| CONCEPT | 203,864,313 | `5be10a0d8900083c9b56cf2e53f17a7e15195879184615512c03690a98da65a4` |
| CONCEPT_RELATIONSHIP | 433,313,452 | `aa2f23b52c5d5f97519b6176469087dad965b7838e73eb62e18160e3696736b8` |
| CONCEPT_ANCESTOR | 174,318,785 | `5e97d2d88d0ffac721e914e683d2af8c92c57721b0415f03377bc9c9f67a29e5` |
| VOCABULARY | 3,710 | `1f5e1fa97c8be54d9f5afeb7d673d5040952c0e9318db125957aa71209e7eb6a` |

The local JSON also records source-file hashes, configuration, dirty-state
description, processor fingerprints, ordered mapping digests and all vocabulary
metadata rows. Licensed input tables and release-specific source/target/path
rows remain in local artifacts; they are not copied into this report.

## Measured graph differences

| Quantity | Count |
| --- | ---: |
| `|U|` | 92,676 |
| Eligible targets, both modes | 69,445 |
| Deep sources, depth > 5 | 23,230 |
| Excluded root, depth 0 | 1 |
| `|D|` | 206,783 |
| `|P|` | 207,117 |
| `|D ∩ P|` | 206,783 |
| `|P − D|` | 334 |
| `|D − P|` | 0 |
| `|D ∪ P|` | 207,117 |

The independently extracted public-reader sets agree with factory direct graph
edges, eligible codes, and provenance counts. Factory
`closure_only_edge_count = 334 = |P − D|` for both policies. The additional
edges amount to 0.162% of the direct edge count. That fraction alone is not
a criterion for removing the option.

## Measured mapping differences

Target sets are compared separately from their stable lexicographic ordering.
All observed ordering changes also change sets; no source loses a target.

| Outcome | first_retained | all_retained |
| --- | ---: | ---: |
| Evaluated sources, all U | 92,676 | 92,676 |
| Identical target sets, all U | 92,624 | 92,611 |
| Changed nonempty target sets | 26 | 39 |
| Direct `no_target` recovered by closure | 26 | 26 |
| Any changed target set | 52 | 65 |
| Lost mappings / other target-change categories | 0 / 0 | 0 / 0 |
| Eligible identity mappings, either mode | 69,445 | 69,445 |
| Direct rolled-up mappings | 23,197 | 23,197 |
| Closure rolled-up mappings | 23,223 | 23,223 |
| Direct mapped / unmapped, all U | 92,642 / 34 | 92,642 / 34 |
| Closure mapped / unmapped, all U | 92,668 / 8 | 92,668 / 8 |
| Deep denominator | 23,230 | 23,230 |
| Identical target sets, deep | 23,178 | 23,165 |
| Direct `no_target`, deep | 33 | 33 |
| Closure `no_target`, deep | 7 | 7 |

All unmapped results inside `U` are `no_target`; `unknown_code`,
`outside_domain`, and `unreachable` each have count zero. The all-U failure
denominator includes the excluded Disease root, whose empty ancestor set is
expected. Thus closure recovers 26/33 = 78.8% of direct **deep** failures,
while recovering only 26/23,230 = 0.112% of all deep concepts. Any changed
target set affects 0.224% of deep concepts under first-retained and 0.280%
under all-retained. These percentages use concepts, not diagnosis events or
patients, as their unit.

## Structural gaps are different from mapping failures

The structural results are identical across target policies.

| Structural metric | Direct total | Direct eligible / deep | Closure total | Closure eligible / deep |
| --- | ---: | ---: | ---: | ---: |
| Non-root nodes without an incoming parent edge | 36 | 23 / 13 | 11 | 8 / 3 |
| Nodes unreachable from Disease along selected edges | 190 | 77 / 113 | 133 | 46 / 87 |

In closure mode, 133 nodes lack a selected path from Disease, but only seven
deep sources fail to map. An eligible disconnected source maps to itself. A
deep source can reach an eligible ancestor in its own disconnected branch.
Neither parentlessness nor root reachability substitutes for typed mapping
results. The excluded root contributes to mapping failures, not these gap
counts.

An additional diagnostic scan filtered `CONCEPT_ANCESTOR` to the seven residual
deep source IDs and the fixed eligible target IDs, then took each source's
minimum recorded separation. All seven have eligible ancestors: the nearest
separation is three for three sources, four for three, and five for one.
They nevertheless have no selected path to a target. Thus the available
longer ancestor relations and the separation-one traversal overlay are
materially different. This does not prove that the current orphan-rescue
algorithm would repair all seven; that remains a separate experiment.
Exact rows are in local `residual_diagnosis.json`.

## Synthetic evidence and a planning correction

The default notebook and ten focused tests use invented Athena-format tables.
Their independently specified expectations are `|U|=12`, `|D|=9`, `|P|=11`,
and two closure-only links. Both policies recover one deep failure. An added
eligible parent changes two nonempty first-retained results, while those
all-retained results already contain the added target. A filtered Observation
requiring a longer shortcut remains `no_target` in both modes; increasing the
cutoff to make that source eligible gives identity mapping in both modes.
The fixture deliberately stresses supplied depths that disagree with direct
adjacency and is not a complete simulated Athena release.

The planning handoff suggested that merely adding an edge could replace an
existing first-retained target at fixed eligibility. **That is incorrect for
these semantics.** Every original path to a first-retained target remains,
with the same noneligible internal nodes. Adding edges does not insert a node
into that old path. Both policies are target-set monotone when acyclic edges
are added and eligibility stays fixed. The tests exhaust the 64 possible
overlays on a four-node ordered DAG for both policies, in addition to the
multi-parent fixture; the full release also satisfies this invariant. Changing
eligibility, deleting edges, or selecting only globally nearest targets would
be different operations.

## Value of the resulting output

Closure mode changes the **derived PyHealth traversal and roll-up output**;
it does not edit Athena's exported tables. The subtree and eligible target
vocabulary stay identical. The factory's base KnowledgeGraph still contains
direct edges in both modes. The processor adds the overlay to its traversal
index, and its per-sample projected graphs can include those added edges with
the `closure_only` mask. Exact local witnesses identify recovered paths and
new target branches. Full-population `rollup_codes()` evaluation does not build
one giant projected graph.

For validation, this is a distinct policy sensitivity dimension: compare
mapping coverage, target sets, and projected-path provenance at the same
depths. A downstream pipeline that consumes changed roll-up targets may change
its histories or aggregate counts; that propagation was not measured here.
The result does not choose an embedding-training graph, quantify co-occurrence
changes, establish clinical usefulness, or reproduce the authors' unpublished
graph construction. Keep reconstruction agreement separate from author parity.

Retaining an explicit option is supported because it has demonstrated behavior
and recovers otherwise dropped sources. A single bundle and unweighted concepts
do not justify choosing a universal default. Orphan-rescue policy remains a
separate comparison; seven deep failures persist. Future work can compare
rescue off/on and, if authorized aggregate frequencies are supplied, report
diagnosis-weighted effects under an explicit counting contract. No MIMIC input,
embedding training, co-occurrence ablation, or production change was performed.

## Reproduction and validation

From the repository root, using an environment with PyHealth's existing
dependencies:

```bash
python -m examples.keep_closure_mode_audit --output-dir .artifacts/c0015-synthetic
python -m examples.keep_closure_mode_audit --athena-dir /path/to/authorized/export --output-dir .artifacts/c0015-athena
python -m pytest tests/core/test_keep_closure_mode_audit.py tests/core/test_ancestor_rollup_processor.py tests/core/test_ancestor_rollup_from_omop.py tests/core/test_omop_vocabulary.py tests/core/test_omop_condition_hierarchy.py -q
```

`examples/keep_closure_mode_audit.ipynb` executes the synthetic explanation by
default. Its optional Athena cell has `ATHENA_DIR = None` until explicitly set.
The tracked notebook has cleared outputs. It was executed in a fresh Python
kernel, with real input disabled, and its assertions passed. The full release
was run through the same helper via the CLI, separately from the notebook.

The focused and existing suites passed **140 tests and 8 subtests**. Ruff
checks and formatting passed for the new Python files. The production loading
test added by the loader optimization does not exist at this baseline; no result is claimed for
that absent suite. An existing PyTorch deprecation warning is unrelated to the
audit. Notebook execution required a local ipykernel runner; no project
dependency file or shared environment was changed.

A second full-release run used batch size 137 instead of 512. All non-resource
results matched after excluding per-phase timings, dirty-state text and batch
size; processor fingerprints and full ordered mapping digests matched. All four
difference/failure TSV files were byte-identical. The comparison record is local
`.artifacts/c0015-determinism.json`. Timing and process-memory observations are
not claimed to be deterministic.

The output directory contains `summary.json`, per-policy `*_differences.tsv`
(changed sources), and `*_failures.tsv` (every failed source, including the
excluded root). Summary witnesses are bounded to three per category. Absent
reason keys in aggregate JSON mean zero. `ordered_mapping_sha256` covers every
source, depth, reason and ordered target tuple, including unchanged sources.
Keep these release-specific artifacts local pending sharing review.

Resource reporting includes the public reference reader's full uncut ancestor
materialization separately from each factory build and each complete mapping
pass. The baseline factory also materializes full ancestry on every build.
Timings cover only the named phases; input hashing, graph assertions, output
serialization and witness projection are not included in mapping time.
Process peak RSS uses `getrusage(RUSAGE_SELF)` and is a lifetime high-water mark
across all phases, not an isolated factory or per-mode measurement. These are
single-process observations, not a controlled benchmark or loader-optimization performance
comparison. See local JSON for exact timings and memory.

| Measured phase, main completed run | Seconds |
| --- | ---: |
| Independent public-reader extraction | 0.447 |
| First-retained direct / closure factory build | 1.039 / 1.118 |
| First-retained direct / closure full mapping | 0.103 / 0.208 |
| All-retained direct / closure factory build | 1.030 / 1.114 |
| All-retained direct / closure full mapping | 0.296 / 0.401 |

Lifetime process peak RSS was 2,217,984,000 bytes (about 2.07 GiB). Other
validation work ran concurrently, so these timings provide resource context,
not a reliable relative-speed estimate. An earlier attempt was interrupted
after audit-side ID decoding repeatedly copied the lookup table; the helper
now captures that lookup once. No interrupted-run measurement is reported.

See [the Kaizen handoff](graph_closure_validation_handoff.md) for the portable finding,
validation integration contract and a draft review reply.
