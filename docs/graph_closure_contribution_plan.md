# Graph closure analysis: PyHealth contribution plan

Prepared 2026-11-25. This standalone plan extends the existing example helper,
notebook and semantic tests. It specifies future implementation; none of the
new experiments below has been run. Measured findings are in
[the evidence report](keep_closure_mode_audit.md). The documentation is prepared for circulation alongside the example implementation.

## Problem, contribution and boundaries

Ontology filtering can leave different traversal graphs depending on whether
the consumer uses direct relationships alone or also uses separation-one
ancestry pairs. Users need to know whether that choice changes their generated
graphs, drops source concepts, or changes target attribution.

Contribute a reusable, reproducible graph-construction validation workflow to
PyHealth: an example helper/CLI, a synthetic-first tutorial notebook, regression
fixtures and an explanation of graph and coverage outcomes. The paired
direct/closure experiment is its central comparison. A useful software
contribution does not require claiming a novel closure algorithm or a better
predictive model. Default policy selection remains a separate decision.

Preserve production APIs/defaults, authoritative root depths, identifier
conventions and existing work. Keep initial implementation in
`examples/keep_closure_mode_audit.py`, its notebook and
`tests/core/test_keep_closure_mode_audit.py`. Do not create a generic production
diagnostics API until another concrete consumer justifies it. Do not train
embeddings, ingest clinical datasets, change orphan rescue, or implement a
co-occurrence ablation as part of this plan. Do not create/switch branches,
commit, push, or open/update a PR without the human's instruction.

## Why graph scope matters for learning

The central question is what information and structure closure introduces into
the learning input. Three effects must be measured separately: previously
unmapped observations contribute graphs; additional targets broaden conceptual
scope; and additional successful paths change topology even when targets stay
identical. The existing mapping census establishes the first two on one bundle;
population-wide generated-graph effects remain to be measured.

For example, direct traversal can produce `T → A → X`, while the overlay adds
`T → X`. If `T` is the retained target, both modes emit the same target for `X`,
but the generated graphs differ. Depending on message direction and architecture,
the shortcut can change neighborhoods, degrees and the number of layers needed
to exchange information. In a GCN, adjacency and degree normalization enter the
computation directly; see the [original GCN paper](https://arxiv.org/abs/1609.02907).
A graph change therefore need not change mapping coverage to affect a learner.

Greater connectivity is not automatically beneficial. It may restore useful
context or introduce additional shared context that reduces discrimination.
These are learning hypotheses, not consequences established by fewer mapping
failures. The factory's unchanged base KnowledgeGraph also means that a separate
embedding trainer consuming only that graph would not receive the overlay
unless explicitly configured to do so.

Assess conceptual scope (targets and intermediate concepts), structural scope
(paths, degrees and model-relevant neighborhoods), and exposure (frequency of
changed graphs in an authorized intended sample population). A small fraction
of vocabulary edges does not determine the frequency or importance of affected
samples. Shared learned parameters also allow retraining to change predictions
for samples whose own graphs are unchanged.

A subsequent learning study requires a separate scope: retain identical cohorts
and train/validation/test splits, match initial features and model settings,
repeat training over paired random seeds, and explicitly handle unmapped or
empty-graph samples in both conditions. Do not let selective sample dropping
masquerade as improved prediction. First isolate sample-graph effects with
shared initial embeddings; retraining embeddings on different graphs is a
separate intervention. A frozen-model input swap can demonstrate sensitivity,
but does not establish a benefit from training with the alternate representation.
No learning study or sample-exposure measurement has been performed here.

## Fixed graph contract

Use the uncut, valid standard SNOMED Condition Disease subtree `U`, direct
parent-to-child `Is a` edges `D`, and unique separation-one ancestry pairs `P`
with endpoints in `U`. Direct selects `D`; closure selects `D ∪ P`. Hold
Athena minimum root depths fixed; the cutoff controls target eligibility,
not which sources/intermediate nodes exist. Direct provenance wins on overlap.

Distinguish three graph layers in every output:

| Layer | Meaning and expected effect |
| --- | --- |
| Factory base KnowledgeGraph | Direct edges and shared entity IDs in both modes |
| Processor traversal index | Direct edges versus the separation-one overlay |
| Generated sample graph from `process()` | Union of successful mapping paths for supplied source codes; may change even when target sets do not |

OHDSI documents `CONCEPT_ANCESTOR` as a derived ancestry table with a minimum
hierarchical separation field. Its ancestry-generation inputs include
`RELATIONSHIP`; it should not be assumed identical to a consumer's filtered
`Is a` subgraph. The existing option uses separation exactly one, not the full
transitive closure. See the [official OMOP table definition](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_ancestor).

## Priority 1: complete graph-output and failure measurement

At cutoff five, excluded root and rescue disabled, evaluate both target
policies separately. Reuse the existing all-U mapping census, input identity,
typed failure counts and deterministic ordering. Extend it as follows.

**Generated singleton graphs.** For every source in `U`, call `process([source])`
in both modes, stream aggregate statistics and discard graph tensors after
comparison. Decode local node indices through `node_ids` and the processor's
entity map before comparing. Do not compare local indices or fingerprints as
if they were graph contents. Record each category's all-U and deep denominator.

| Measurement | Required output |
| --- | --- |
| Graph contents | Node and edge counts in each mode; decoded added/removed node and edge sets; seed/target set changes; closure-only edge counts |
| Graph change versus mapping change | Cross-tabulate graph changed/unchanged against target set changed/unchanged, including recovered failures |
| Size and amplification | Distribution of node/edge deltas: median, p95, maximum; source witness for maxima; counts of empty and edgeless nonempty graphs |
| Target multiplicity | Distribution and change in target counts, separating identity mappings from deep-source roll-up |
| Productive overlay use | Count distinct added traversal edges observed on at least one successful singleton path, over `|P−D|`; do not equate an unused edge at one cutoff with universal redundancy |
| Graph contract failures | Out-of-range indices, inconsistent edge/feature/mask lengths, wrong decoded IDs, provenance disagreement, missing seeds/targets, cycles or unexpected graph-content removals |

Report graph changes for unchanged target sets explicitly: an alternate
successful path may change graph input while leaving mapping coverage intact.
An eligible identity source yields a nonempty graph with no path edges; do
not count this as graph generation failure. An unmapped source yields an
empty graph under diagnostic `on_unmapped="empty"`; count it as a mapping
failure, not as a malformed graph. The same source's default-error exception
is the operational manifestation of that failure, not an additional event.

**Composed sample graphs.** Add hand-specified synthetic multi-source/visit
fixtures: shared ancestors, duplicate observations, mixed mapped/unmapped
sources, recovery, unchanged-target/changed-path cases, and all-unmapped input.
Compare decoded union graphs and masks against independent expected sets.
Duplicate observations must remain duplicated in `rollup_codes()` results but
must not duplicate graph entities/edges. Check feature rows using distinctive
synthetic vectors and verify optional PyG batch/unbatch round trips preserve
IDs, edges, features and masks. Optional PyG checks must be visibly skipped
when the dependency is absent. See [PyG batching conventions](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html).
These constructed samples establish composition behavior; they do not estimate
patient-level graph frequencies.

**Define the rates.** Primary mapping failure rate is deep `no_target` sources
divided by all deep sources in the same configuration. Also report all-U
counts and eligible/excluded-root strata, recovered-failure fraction, changed
nonempty target fraction, graph-contract violation counts, and construction
exceptions separately. For this measured bundle the deep mapping failure rate
is 0.142% direct versus 0.030% closure. This is a 0.112 percentage-point absolute
decrease; 78.8% is the relative recovery fraction. It is not semantic accuracy.
Do not estimate a failure rate only from saved difference rows or selected
witnesses. External-code failures need a separate, explicitly defined input
population; they cannot be inferred from an all-U census.

## Priority 2: independent reference and cause attribution

The current reference and factory share the public OMOP reader. Their agreement
checks integration, but cannot exclude a shared extraction bug. Add two
independent checks before presenting the workflow as a complete validator:

1. For tiny coherent synthetic Athena tables, implement a test-only reference
   using explicit edge sets and straightforward path enumeration on a DAG.
   Compute policy-specific targets, successful-path edges and masks without
   calling the production traversal or reading its private indexes. Keep the
   existing adversarial supplied-depth fixture separately labeled. Include a
   case with identical targets but different successful-path graphs.
2. Independently extract root depths, validity/domain-filtered direct edges
   and separation-one pairs using standard-library CSV parsing or another
   existing engine without invoking `condition_hierarchy()`. Check exact sets,
   orientation, duplicates and root self-row handling against the factory on
   synthetic input, then compare sorted set digests on the authorized release.
   Keep memory bounded and record extraction cost. Do not add a dependency
   solely for this check.

For each closure-only edge and remaining failure, retain local diagnostic
evidence identifying endpoint domains/standardness/validity, whether a valid
direct relationship exists before filtering, and recorded longer ancestry to
eligible targets. Classify causes only where the tables establish them;
otherwise label them unresolved. If an explicitly available `RELATIONSHIP.csv`
supports checking other ancestry-defining relationships, record that input's
identity too. Do not infer Athena's entire generation algorithm from a missing
`Is a` row. The observed seven residual deep failures have eligible ancestors
at separation three to five; that fact alone does not show the current rescue
algorithm would repair them.

Coverage and synthetic correctness are separate evidence axes. Additional
reachable targets on real data are not automatically correct clinical targets.
No precision/recall or semantic error rate is available without an independently
justified reference labeling task.

## Priority 3: bounded sensitivity and robustness

After the primary graph and reference checks pass, run the predeclared cutoffs
3, 4, 5, 6 and 7 with both target policies. Within each pair preserve the same
uncut `U`, root depths and rescue-disabled setting; eligibility and the deep
denominator change across cutoffs and must be recomputed. Report coverage,
graph deltas and target multiplicity for every cell, including zero effects.
Use root-inclusion and cutoff-zero cases as synthetic boundary checks, rather
than silently mixing their failures with the primary excluded-root analysis.

Add controlled synthetic perturbations with known effects: remove a direct
link while retaining its separation-one pair; remove both forms; filter an
intermediate requiring a longer shortcut; duplicate/reorder rows; and inject
a self-edge/cycle. Recovery is expected only where the contract supplies an
alternative path. Invalid graphs must fail explicitly. If varying deletion
fractions/seeds, label it a stress test whose missingness mechanism is chosen,
not an estimate of real Athena defect rates.

An additional authorized release is a useful replication extension, not a
requirement for a bounded single-release contribution. Recompute each release's
own populations and metadata; compare shared IDs separately from additions and
removals. No current result establishes cross-release robustness.

This is a census of a fixed vocabulary graph, with correlated ancestry. Report
exact counts and paired deltas. Do not attach independent-concept confidence
intervals or p-values as though concepts were a random independent sample.
Clinical event or patient weighting requires a separately authorized input
and counting contract; it is not part of the minimum contribution.

## Resource measurement and maintainable delivery

Keep mapping-only and graph-projection costs separate. Record reference
extraction, factory construction, full mapping and full projection durations,
graph-size distributions and the memory measurement method. Isolate runs if
making performance comparisons; the existing concurrent-run timings are only
context. Retain bounded witnesses, stream summaries, and avoid building a
single graph for the entire vocabulary. A process-wide peak is not a
per-mode memory estimate. Recheck the intended upstream base: the measured
baseline predates the separate loading optimization.

Extend existing files with reusable graph comparison and summary helpers;
avoid a second implementation embedded in the notebook. CI uses synthetic
tables with hand-specified expectations. Keep full Athena execution opt-in,
with no path discovery or data downloads. Keep licensed rows local. Upstream
titles, notebook prose and comments describe graph closure behavior and
coverage, with no internal contribution IDs or links to unpublished docs.

## Acceptance and contribution description

Minimum next milestone: Priority 1 and Priority 2, plus the bounded cutoff
table from Priority 3. Synthetic checks must validate exact graph contents,
composition, typed failures, independent extraction, provenance and fixed-
eligibility monotonicity. On the supplied bundle reproduce the existing edge
and mapping counts, then measure the newly specified graph results without
assuming their values. Repeat non-resource summaries deterministically and
execute the notebook in a fresh kernel with outputs cleared. Document all
skips and separate the old measured results from newly completed extensions.
Do not mark this milestone complete from the present mapping census alone.

Suggested upstream title: **Validate graph closure effects on generated graphs
and mapping coverage**.

Proposed description once those checks exist:

> Adds a reproducible workflow for comparing direct ontology traversal with
> separation-one ancestry augmentation. It reports generated graph changes,
> mapping failure rates and edge provenance under matched settings, with
> independent synthetic references and opt-in Athena evaluation. It helps
> users validate graph preprocessing choices while preserving existing
> production behavior and defaults.

Limit the actual PR description to methods implemented and checks actually
run. Until the extension milestone is complete, describe the current delivery
as a graph closure comparison example with mapping-coverage measurements and
selected projected-path witnesses. Neither stage claims predictive benefit,
author-artifact parity, guaranteed orphan repair, or a new default policy.
