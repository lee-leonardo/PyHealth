# Graph closure analysis: validation handoff

Prepared 2026-11-25 from the supplied PyHealth worktree at production revision
`33c2cb635572a493dae428c4666a5ec503c5ee77`. This document is ready to transfer;
the Kaizen and validation repositories have not been edited by this task.
The complete measured report is [keep_closure_mode_audit.md](keep_closure_mode_audit.md).

## Contribution direction

The human clarified the intended contribution: validate how the separation-one
closure behavior affects **graph generation and mapping failure rates**.
Position the paired comparison as one component of a reusable PyHealth graph
validation workflow. The standalone [graph closure contribution plan](graph_closure_contribution_plan.md)
specifies implementation boundaries, graph metrics, an independent reference,
error definitions, sensitivity checks and acceptance criteria. These extensions
are proposed, not implemented or measured. Carry this framing into Kaizen and
the validation project; keep internal contribution IDs out of upstream artifacts. The contribution plan also records the agreed distinction
between mapping coverage, graph scope and downstream learning impact.
The current evidence supports a bounded example/validation contribution;
it does not establish a new closure algorithm or a semantic accuracy improvement.

## Finding to record

On the supplied Athena bundle (`v5.0 29-AUG-26`; SNOMED International
2026-02-01 / US 2026-03-01 / UK 2026-04-08), adding minimum-separation-one
ancestry pairs to valid direct Disease/Condition edges adds **334 edges** to
206,783 direct edges over 92,676 concepts. At fixed minimum root depths,
cutoff five, excluded root, and disabled orphan rescue, both target policies
recover **26 of 33 deep `no_target` sources**, leaving seven deep failures.
First-retained changes a further 26 nonempty target sets; all-retained changes
39. There are 23,230 deep sources and 69,445 eligible targets; target eligibility
is identical across modes. No mapping or target is lost.

Recommendation: retain the explicit closure option based on demonstrated
mapping coverage and target differences on this bundle. Leave default choice
and orphan-rescue policy open. This is vocabulary behavior, not a measured
clinical/diagnosis-weighted benefit or embedding-quality result.

## Evidence and artifacts to carry

| Artifact | Use |
| --- | --- |
| `examples/keep_closure_mode_audit.py` | Reusable helper/CLI, synthetic default and explicit real input |
| `examples/keep_closure_mode_audit.ipynb` | Executed synthetic explanation; outputs cleared; real run opt-in |
| `tests/core/test_keep_closure_mode_audit.py` | Ten independent semantic checks |
| `docs/keep_closure_mode_audit.md` | Methods, input hashes, denominators, measured results, limits |
| This document | Contribution update and standalone validation integration plan |

Local `.artifacts/c0015-athena/summary.json` records complete release metadata,
input and source hashes, configuration, processor fingerprints, ordered mapping
digests, timings, memory, and bounded exact path witnesses. Per-policy difference
and failure tables retain source IDs, depths, typed reasons, ordered targets,
added/removed target sets and categories. These rows and the vocabulary bundle
stay local pending sharing review. Aggregate findings and hashes are in the
report; do not copy licensed tables into Kaizen.

Validation performed: 140 tests plus 8 subtests passed, including the ten new
checks and existing OMOP vocabulary/hierarchy and roll-up suites. The notebook
passed in a fresh kernel with Athena disabled. Full-release paired evaluation
completed for both policies. A second full-release run with batch size 137
instead of 512 reproduced non-resource results, processor fingerprints and full
ordered mapping digests; all four local difference/failure TSVs were
byte-identical. The comparison excludes timing, peak RSS, dirty-state text and
batch size and is recorded in local `.artifacts/c0015-determinism.json`.
This checkout predates the separate loading optimization; no performance or loading
test result from that other revision is claimed here.

## Corrections for the contribution record

Mark the synthetic tooling, measured release comparison and bounded evidence
report complete once these artifacts are accepted. Replace the prior
"unresolved Athena input" note: the human supplied a local bundle and its
entire Disease subtree was evaluated. Keep validation-project integration
separate from completion of the audit; it has not been implemented there.

Correct the old implementation handoff's first-retained replacement claim.
Adding acyclic edges with fixed eligibility preserves old target sets under
**both** policies. An original first-retained path remains intact. The audit
checks this with a multi-parent fixture, all 64 overlays of an ordered
four-node DAG for each policy, and every source on the measured release.
Do not add a test requiring target removal for this operation.

The actual production baseline here is `33c2cb6`, not the historical `3c86e39`
optimization checkpoint. No production API, defaults, loader location,
orphan-rescue behavior changed in this work. The helper, notebook and tests were
subsequently committed and pushed on `task/closure_mode_investigation`; the
scrubbed published commit is `b2a5150` ("Add KEEP graph closure analysis"). No
open PR was found at the last check. The documentation is now prepared for circulation on the same personal branch.

## Standalone validation-project integration plan

The next validation implementation should treat traversal policy as a
separate, explicitly configured sensitivity analysis. Preserve existing
reconstruction-agreement findings and their evidential limits.

1. Accept an explicitly supplied Athena directory. Verify the report's table
   hashes before claiming to reproduce this bundle; otherwise label the new
   release and retain its metadata/hashes. Do not discover datasets or add
   MIMIC access as part of this task.
2. Run the helper on the pinned PyHealth revision, or port the audit operation
   after confirming the current public interfaces. Preserve the full uncut
   node set and authoritative `min` root depths. Hold cutoff five, excluded
   root, diagnostic `on_unmapped="empty"`, and rescue disabled fixed. Run
   both first-retained and all-retained separately; do not designate either
   as the author-verified policy.
3. Add a notebook/report section named "Athena traversal-policy sensitivity"
   that separates synthetic semantic evidence from full-release evidence.
   Display graph deltas, all-U and deep denominators, typed failures,
   changed nonempty target sets, recovered failures and structural gaps.
   Show that the target vocabulary remains identical. Explain the excluded
   root's additional `no_target` separately from deep failures.
4. Check graph identities `|D ∪ P| = |D| + |P − D|` and processor provenance,
   exact target-set monotonicity, eligible identity behavior, and population
   reconciliation. On this exact bundle, expect `P−D=334`, all-U target-set
   changes `52/65` by policy, and deep failures `33→7`. Keep per-source rows
   local, and link a few reviewed witness paths only if sharing is authorized.
5. Record sample-projection edge provenance independently of target mapping.
   The factory's base KnowledgeGraph remains direct; closure is a traversal
   overlay. Do not silently reuse the overlay as an embedding-training graph.
   No embedding-quality conclusion is available from this audit.
6. If histories or co-occurrences are compared later, derive them from each
   policy's actual emitted targets under an explicit event-counting contract.
   That is downstream work, not evidence already established here. Frequency
   weighting requires a separately authorized aggregate input, with its unit
   and denominator specified. Do not infer prevalence from vocabulary counts.
7. Leave orphan-rescue off/on as a separate future experiment. Never label
   separation-one closure as arbitrary ancestor lookup, guaranteed orphan
   repair, or author-artifact parity. Retain the seven unresolved deep cases
   as explicit failures rather than dropping them from denominators.

Acceptance: the section is reproducible with recorded revision, release,
hashes/configuration, complete population coverage and deterministic outputs;
it distinguishes policy sensitivity from reconstruction agreement and
author parity; it makes no downstream or clinical claim without new evidence.
Further implementation, default/API changes and publication require a separate instruction.

## Draft review reply for the human

On the supplied release, `closure_only_edge_count` is 334 over 206,783 direct
edges and 92,676 uncut Disease-subtree concepts. At cutoff five with fixed
Athena minimum depths, excluded root and no orphan rescue, the overlay recovers
26 of 33 deep mapping failures under both target policies. It also changes
nonempty target sets for another 26 first-retained and 39 all-retained sources;
seven deep failures remain. The 69,445 eligible target codes are unchanged.
This supports retaining an explicit closure option on this bundle; the small
edge fraction alone would miss useful coverage differences. We have not
measured diagnosis prevalence, clinical impact, embedding quality or author
graph fidelity, and this does not decide defaults or orphan-rescue policy.
The report records the input hashes, full denominators, structural gaps and
reproduction steps; exact source/path witnesses remain local.

This is a draft only; it has not been posted or sent.
