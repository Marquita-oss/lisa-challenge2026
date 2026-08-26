# Response to Reviewers — LISA Challenge 2026, Submission 11

*Rigor over Novelty in Ultra-Low-Field Pediatric Brain MRI: Quality Control and Subcortical
Segmentation for the LISA Challenge 2026* (Synapse `syn75277286`)

We thank both reviewers for reviews that were specific enough to act on. Every substantive
request has been addressed with new analysis rather than with argument, and one of the results
runs against a design choice of our own — we report it as such. All new numbers come from the
already-stored out-of-fold predictions; **no test-set or leaderboard information was used**, and
nothing was retrained.

A change-tracked copy of the manuscript accompanies this response: additions and deletions are
coloured **blue for Reviewer 1**, **orange for Reviewer 2**, purple where both asked for the same
thing, and green for one correction we found ourselves.

---

## Reviewer 1 (Atw8)

### R1.1 — "The authors should report positive-case recall by acquisition plane"

Done, and it substantiates the reviewer's suspicion. **New Table 3** and a new paragraph in
Section 3.1 report positive-case recall (the fraction of truly affected planes, severity ≥ 1,
flagged as affected) per acquisition plane, both pooled and per artifact class.

Pooled recall is 0.618 axial, 0.508 coronal and 0.629 sagittal, and exact-grade agreement on
detected positives splits further apart (0.565 axial against 0.362 coronal). Per class the
pattern tracks the geometry of the artifact: zipper is recovered in 0.86 of sagittal positives
but only 0.45 of coronal ones, and banding ranges from 0.17 axial to 0.71 coronal. Positioning
and distortion stay below 0.52 in **every** orientation, which tells us their difficulty is
representational rather than an orientation effect.

We also state two caveats in the text rather than letting the table over-claim: prevalence is not
balanced across planes (110 of the 179 zipper positives are sagittal, against 38 axial), so part
of each gap reflects training exposure; and micro-accuracy runs opposite to recall because it is
dominated by true negatives.

### R1.2 — "The data slice selection for Task 1a is worrisome and does not recognize the locality of certain features"

We took this as the central objection and tested it directly. **New Table 4** reports a
slice-sampling ablation that holds the ten trained models, the fold assignment and the calibrated
thresholds fixed and changes only which slices form the three input channels — including a
sliding three-slice window over the *entire* through-plane axis. The out-of-fold, leakage-free
property is preserved exactly.

The result is genuinely two-sided, and we report both halves:

- **The reviewer is right about localised artifacts.** Exhaustive slice coverage raises zipper
  recall from 0.726 to 0.799 (axially, 0.55 → 0.74) and positioning from 0.357 to 0.400. Fixed
  percentile sampling does miss localised evidence.
- **But the fix is not uniform.** Motion (0.527 → 0.314) and distortion (0.424 → 0.215) collapse
  under dense sampling, consistent with signatures that are inter-slice inconsistencies and are
  washed out by averaging over strongly overlapping triplets. Specificity falls from 0.947 to
  0.940 and net micro-accuracy from 0.839 to 0.826.
- **The three-slice input is nevertheless not arbitrary.** Collapsing to a replicated central
  slice costs 0.024 micro-accuracy and drops pooled positive recall from 0.591 to 0.450.

The conclusion we now draw in the paper is that no single sampling density can serve seven
artifact classes whose evidence lives at different spatial scales, and that a rule conditioned on
the artifact under test is the concrete next step. We also state the limit of the experiment
plainly: all ten models were *trained* on the 25th/50th/75th-percentile input, so this measures
robustness of the learned representation to a change of sampling at inference, not what a densely
trained or fully volumetric model would achieve. A new Discussion paragraph, *"Where our own
design choice does not hold up"* (Section 4), says so without hedging.

### R1.3 — "Fold the ventricles back into the full consideration of the ranking given 2026's updated criteria"

Done throughout. Section 2.3 now states the criteria explicitly and notes that we built and
submitted under the reading that the ventricles were excluded; every Task 2 result is reported
under both aggregations, with the eleven-label ranking leading.

- **Table 8** (per structure) folds the ventricles into the table body, with **Mean (11 labels)**
  as the primary row and the nine-label mean below it for continuity: Dice 0.7842 against 0.7849,
  ASSD 0.920 mm against 0.978 mm.
- **New Table 7** repeats the post-processing ablation under both label sets. The adopted
  structure-specific scheme remains best on DSC, HD95 and ASSD under *both*, and uniform
  largest-component filtering still degrades ASSD (0.920 → 1.125 mm over eleven labels).
- The abstract and contributions now quote 0.784 over eleven labels.

We would add that scoring the ventricles makes this check **stricter**, not looser: they are two
of the three labels the adopted scheme actually filters, so the conclusion survives a harder test
than the one originally reported.

### R1.4 — "The work does not introduce a new architecture or learning method"

Accepted without qualification, and now stated by us rather than left for the reader to notice.
The Introduction says plainly that the paper contributes no new architecture and no new learning
rule, and that what it offers is a rigorously validated baseline together with the negative
results that delimit it — including, per R1.2, evidence against one of our own choices. We would
rather this be legible as a deliberate scope than read as an omission.

---

## Reviewer 2 (d7ui)

### R2.1 — "The three-slice approach needs more justification; compare with a denser multi-slice or 3D representation, or dynamic slice selection"

Section 2.1 now gives the reasoning we had left implicit: three percentile slices buy volumetric
coverage at the fixed cost of the three channels an ImageNet-pretrained backbone already expects,
and avoid the degenerate alternative of replicating one central slice. It also states the
assumption the choice makes, and that the assumption is stronger for localised artifacts than for
global ones.

The empirical comparison is **new Table 4**, described under R1.2 above: single central slice, the
adopted percentile triplet, and three progressively denser schemes up to a sliding window over
every slice. Two things it does **not** do, which we say in the text rather than imply otherwise:
it does not train a densely sampled or fully volumetric model (an inference-only ablation cannot
stand in for that), and it does not implement dynamic slice selection. Both are now named
explicitly as future work rather than described in general terms.

### R2.2 — "Report Task 1a results in more detail: per-artifact performance, variation across folds, effect of design choices"

- **Per artifact.** Table 2 gains a positive-case recall column alongside accuracy and QWK, and
  Table 3 adds the per-artifact breakdown by acquisition plane.
- **Across folds.** Section 3.1 now reports per-fold micro-accuracy: 0.821, 0.838, 0.838, 0.843,
  0.859 (mean 0.840, sd 0.013). We also use it, rather than just report it: differences between
  configurations of about a point are within fold-to-fold variation on a 532-image cohort, and
  the text now reads them that way.
- **Design choices.** Table 1 separates backbone, ensembling, calibration and TTA; Table 4 covers
  the input representation.

### R2.3 — "Phrases such as 'forces hallucinations' are too strong"

Agreed. All three occurrences are reworded, keeping the finding and dropping the causal certainty
the evidence does not support:

| Location | Was | Now |
|---|---|---|
| Abstract | "a registration ceiling … that **makes supervision hallucinate**" | "…that leaves the supervisory target only partially aligned" |
| Section 3.2 | "the model **is forced to hallucinate** high-field texture in slightly wrong locations" | "the model synthesises high-field texture at slightly displaced locations, which the reference metrics penalise" |
| Section 4 | "**makes pixel and feature losses hallucinate**" | "makes pixel and feature losses reward texture which is plausible but spatially misplaced" |

---

## A correction we made ourselves

While rebuilding the Task 1A pipeline to produce the per-plane analysis, we found that the stored
out-of-fold predictions behind the reported 0.839 were generated **without** test-time
augmentation, although Table 1 labelled that row "Ensemble (10 models) + TTA". TTA is real, but it
belongs to the submission path, not to the out-of-fold estimate.

The row is now labelled for what it is, and TTA appears as its own measured row: adding eight-way
TTA gives 0.837, i.e. it does not improve the out-of-fold estimate. Separately, three accuracy
cells and three QWK cells in the per-class table differed by up to 0.02 from what the released
out-of-fold predictions reproduce; they have been recomputed, and the mean is unchanged at 0.839.
We report this because a table promising TTA where there is none is exactly the kind of slippage
this paper argues against.

---

## Manuscript length

The revision adds three tables and roughly a page and a half of analysis, taking the manuscript
from 12 to 14 pages including references. We compressed prose elsewhere to absorb most of the
growth and are happy to trim further if this exceeds the camera-ready allowance; we would prefer
to keep Tables 3, 4 and 7, since they are the substance of the reviewers' requests.

## Reproducibility

Every new number is produced by a script released with the code, from artifacts already in the
repository:

| Result | Script |
|---|---|
| Per-plane recall, per-fold variation, per-class detail (Tables 2, 3) | `paper_combined/analysis/task1a_oof_analysis.py` |
| Slice-sampling ablation (Table 4) | `task_1a/slice_ablation.py` |
| TTA row of Table 1 | `task_1a/slice_ablation.py --tta on` |
| Task 2 under both label sets (Tables 7, 8) | `paper_combined/analysis/task2_label_sets.py` |
