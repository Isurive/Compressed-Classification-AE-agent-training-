# REASON-001 — First-pass experimental design

**Status:** Proposed design for review; no dataset/model choice is finalized and no experiments were run.

**Prepared:** 2026-11-05. **Role:** experimental reasoning agent. **Bootstrap owner:** `/root/experiment_reasoner`, reserved by the coordinator.

**Inputs read:** `Final Project/projectIdea.md` and `Final Project/AI360_Project_Proposal_Template.tex`. The ideas below are reasoning from those project requirements, not new empirical findings. Literature claims must be checked against the paper researcher's evidence before entering the proposal.

## 1. Recommended direction

Study **how much the stored training set can be compressed before a classifier's accuracy degrades**, and whether a genuine neural dictionary module changes that trade-off relative to a conventional neural classifier.

Suggested main question:

> With training-image identities and resolution fixed, does a neural sparse-dictionary classifier maintain more held-out classification accuracy than a comparable conventional classifier when the training images are strongly JPEG-compressed, and how does dictionary size affect that result?

This preserves the user's latest storage motivation and earlier dictionary-capacity interest. The experiment should train on compressed data and evaluate primarily on a **common original held-out test set**. Testing on compressed images can be an additional evaluation, but is not the primary evidence that higher-quality training images can be replaced.

The target of 98% absolute accuracy above 50:1 compression remains an exploratory aspiration. Do not optimize the dataset selection just to manufacture that headline, silently reinterpret 98% as relative accuracy retention, or promise that all compression ratios above 50:1 will work.

## 2. Smallest useful comparison

Start the feasibility pilot with four conditions:

| Model | Training data | Main evaluation | Purpose |
| --- | --- | --- | --- |
| Conventional baseline | Original training images | Original test images | Reference learning performance |
| Conventional baseline | Strongly compressed versions of the same training images | The same original test images | Effect of compression on ordinary training |
| Neural dictionary model, one provisional size | Original training images | The same original test images | Reference for the dictionary architecture |
| The same dictionary architecture/size | Strongly compressed training images | The same original test images | Effect of compression with dictionary learning |

Use one seed for a **pilot only**. Its purpose is to establish correctness, runtime, and whether either architecture learns; do not report the pilot as a statistically established advantage. Do not start by training a separate model for every train-quality × test-quality combination: test-quality sweeps reuse already-trained models.

After the pilot, add two further dictionary sizes if training time permits. A focused main study with one conventional baseline plus three dictionary sizes, two training conditions, and three seeds requires:

`(1 baseline + 3 dictionary sizes) × 2 training conditions × 3 seeds = 24 training runs`.

A moderate-compression training condition raises this to 36 runs. A second severe target raises it further. Estimate the total from measured per-run time before choosing the grid. For illustration, 24 runs at 30 minutes require 12 hours; at four hours they require 96 hours, before tuning and failed runs. These are arithmetic examples, not measured hardware estimates.

Three seeds and the illustrative atom counts 64/128/256 are suggestions. The user has not selected these settings. If resources only permit two sizes, report the narrower study honestly rather than running many undertrained configurations.

## 3. A meaningful neural dictionary component

Changing CNN channel count alone does not establish dictionary learning. The model should expose a learned dictionary `D` whose atoms reconstruct a representation through sparse coefficients `a`, with `x ≈ D a`.

### Option A: sparse patch encoder with a linear reconstruction dictionary

This is a relatively transparent implementation candidate:

1. Decode stored JPEG bytes and extract a fixed patch representation from the compressed image.
2. A neural encoder produces `M` coefficients for each patch.
3. Apply a top-k operation to the coefficient magnitudes, retaining at most `k` nonzeros per patch. State whether coefficients are signed or nonnegative.
4. Learn a linear decoder `D` with `M` normalized columns to reconstruct the patch from its coefficients.
5. Pool the sparse codes, retaining a fixed amount of spatial layout if needed, and predict the content class with a small classifier.
6. Train with classification cross-entropy plus a weighted patch-reconstruction objective.

The reconstruction target in the compressed-training condition must be the **decoded compressed patch**, not the original higher-quality patch. Otherwise the experiment requires access to the supposedly replaceable original training images. The originals may be used to construct the compressed dataset before training; they must not remain hidden supervision during that condition.

The reconstruction coefficient, patch dimensions/stride, pooling, dictionary normalization, and optimizer schedule are fixed or chosen using validation data. Track atom usage and reconstruction behavior to verify that the dictionary is functioning rather than merely carrying a label.

A learned feature extractor before the dictionary is also possible. However, reconstructing a jointly learned feature vector introduces extra ways for the encoder to make its own reconstruction problem trivial. If choosing that variant, specify how feature scale/collapse is controlled and compare against the same feature extractor without the dictionary.

### Option B: an unrolled sparse-coding network

An alternative is a short learned sequence of sparse-inference updates, followed by classification. Its dictionary should be explicitly connected to the updates/reconstruction. Depth and sparsity behavior need their own controls. A soft threshold or L1 penalty will not guarantee exactly `k` active atoms; measure the achieved active count rather than claiming it is fixed.

Choose this option only if a suitable implementation is understood and reproducible early in week one. Do not simultaneously build a complicated new backbone, optimizer, codec, and sparse-inference algorithm.

### Controls needed for either option

- Changing `M` changes parameter count and possibly classifier-head size. Report those changes and avoid presenting capacity gains alone as proof that sparsity caused robustness.
- Keep depth, patch geometry, training steps, and active-atom definition constant across the primary size comparison.
- With top-k coding, define the budget **per patch/location**. `k` active at each location is not `k` atoms used across the entire image.
- A conventional baseline trained with the same compressed inputs is essential. An optional dense bottleneck of comparable width is a useful ablation if resources permit.
- A matched-pipeline dense control with the same reconstruction objective would isolate sparsity more directly; without it, describe the finding as an advantage/disadvantage of the full dictionary-based design rather than of sparsity alone.
- The dictionary parameters are part of model storage, not compressed training-file storage. Report both separately.

## 4. Compression and storage protocol

Specify the encoder, chroma-subsampling choice, metadata handling, resolution, bit depth, color representation, and whether original files were already lossy. Generate every tested compressed variant from the same reference representation, avoiding accidental compression chains.

For an 8-bit RGB image `i`:

`raw_bytes_i = 3 × W_i × H_i`

`bpp_i = 8 × jpeg_file_bytes_i / (W_i × H_i)`

`R_raw_i = raw_bytes_i / jpeg_file_bytes_i = 24 / bpp_i`.

Thus 50:1 relative to raw RGB corresponds to 0.48 bpp. A JPEG quality factor of 50 has a different meaning.

For the practical storage claim, measure aggregate quantities:

`R_existing = sum(reference_file_bytes_i) / sum(jpeg_file_bytes_i)`

`saving_percent = 100 × (1 - sum(jpeg_file_bytes_i) / sum(reference_file_bytes_i))`.

Use the actual stored image bytes and state whether unchanged labels/manifests are included. Do not average per-image compression ratios and call the result the aggregate storage ratio. Report the rate distribution as well, because a shared quality setting does not create a shared per-image bit rate.

The pilot should determine whether the intended targets are attainable under **whole-file accounting**. For tiny images, a large fraction of a JPEG file may be headers; a target based only on an estimated entropy payload is a different storage definition. Do not inflate the apparent compression by upsampling a small source image. If a rate is unattainable, report the achieved minimum rate and coverage rather than dropping difficult images from the comparison.

Practical starting approach: use a small fixed JPEG-quality grid, measure the resulting stored bytes, then select meaningful attainable storage levels on training/validation data. If the study instead targets per-image byte budgets, document the quality-selection procedure and how it deals with unattainable budgets.

## 5. Outcomes and what they mean

Primary outcomes:

- Top-1 accuracy on the same original held-out test set.
- Actual stored training bytes and aggregate savings relative to the reference dataset.
- Accuracy difference from the relevant original-training condition.
- Model parameter count and model-file bytes.

Let `A(m,r)` denote accuracy for model `m` trained at storage/compression condition `r`, evaluated on the common original test set. Define:

`drop(m,r) = A(m,original) - A(m,r)`.

A useful interaction quantity is:

`drop(baseline,r) - drop(dictionary,r)`.

A positive value suggests the dictionary design lost less accuracy in that tested setting. Always show absolute accuracies alongside it: a weak model can have a small drop simply because it already performs poorly. Compare same-seed runs where possible and retain per-example predictions for paired analyses. Report across-seed mean and variation; a small number of seeds does not justify broad significance claims.

Choose any acceptable loss tolerance on validation data before final testing, such as one or two percentage points if justified by the pilot and course scope. Define the largest **tested** compression or smallest **tested** dictionary meeting the tolerance; do not imply an optimum over all possible settings.

Claim supported by a successful experiment:

> On dataset X, using fixed image resolution, sample count, training budget, and evaluation protocol, training on JPEG files with Y% fewer stored bytes achieved Z% accuracy, a change of D percentage points from the reference. The dictionary design changed this trade-off by the measured amount relative to the baseline.

Claims not established by this study alone: universal replacement of image datasets; guaranteed 98% at arbitrary high compression; mathematically proving that compression loses no important information; faster compute or lower GPU memory solely because disk files are smaller; novelty of the broad idea of compressed-data training.

## 6. Leakage, initialization, and fairness controls

1. Split by original image identity before making variants. Group related images if the dataset contains multiple images of the same source/subject.
2. Keep test labels out of encoder-quality selection, tolerance selection, architecture selection, and early stopping.
3. Hold image count and label distribution fixed across storage conditions. Equal-storage comparisons with fewer original images answer a second question; defer them unless the core study is complete.
4. Audit the compressed-training loader so it cannot read original pixels, original-image embeddings, or original-image reconstruction targets during training.
5. Use equal step budgets and comparable hyperparameter search effort. Record preprocessing and augmentation order; a resize or recompression after decoding can change what intervention is actually being measured.
6. Random initialization of a reused architecture directly tests training from the selected compressed data. If pretrained weights are used, use the same pretraining source across comparable models and narrow the storage conclusion to the fine-tuning dataset.
7. Do not compare a pretrained model to a randomly initialized model and attribute the result to dictionary learning.
8. Retain the untouched original test set for the primary storage conclusion. Additional compressed-test results should be labeled separately as deployment robustness.

## 7. Four-week execution proposal

| Stage | Bounded work | Exit evidence |
| --- | --- | --- |
| Week 1 | Confirm closest papers, dataset and compute access; choose one model route; build a small compression pilot | Achievable byte sizes, split manifest, one end-to-end learning run, measured runtime |
| Week 2 | Complete controlled reference/compressed pilot for both architectures; repair methodology problems | Four comparable pilot conditions and decision on a feasible final grid |
| Week 3 | Run the frozen main grid with repeated seeds; collect storage and prediction records | Completed run manifests, metrics and failure records; no test-driven retuning |
| Week 4 | Analyze interactions, inspect failures, perform one targeted ablation if justified, write report/slides | Reproducible accuracy–storage comparisons, limitations, and clearly scoped claims |

Proposed human/agent division: data/compression and reproducibility; model/dictionary implementation; literature and evaluation. These are suggestions, not assignments of named team members. The course proposal requires a one-paragraph introduction, a maximum of four pages excluding references, and completed team/timeline tables.

## 8. Decisions still requiring evidence or user information

- Which dataset/task is meaningful and sufficiently challenging for the user's intended claim?
- What resolution is native or defensible, and are >50:1 full-file targets attainable?
- What compute is actually available and how long does one run take?
- Does the dictionary model learn adequately before compression is introduced?
- Are pretrained weights needed, and would the resulting fine-tuning-only claim satisfy the user?
- Which atom sizes, active counts, and reconstruction weights fit the task?
- Does the closest prior work already contain this interaction analysis?
- What empirical extension is sufficient for the instructor's originality expectation?

**Immediate recommended next task:** a feasibility specification selecting candidate datasets and defining the byte-accounting/training pilot, using literature and actual hardware evidence. Do not launch the full training grid from this reasoning memo.

## 9. Handoffs

**To the paper researcher:** verify which closest papers train on compressed data versus merely evaluate compressed inputs; record whether compression is counted against raw pixels or existing files; identify dictionary-size sweeps, pretraining conditions, and original-target supervision. The most useful novelty comparison is at the level of experimental intervention and claim, not overlapping title keywords.

**To the implementation agent:** use a small explicit dictionary module, first establish learning on the reference data, then confirm the compressed-only loader's provenance. Produce a configuration/run manifest before large runs. Every metric must identify train compression, test compression, model size, seed, training budget, and storage denominator.

**To the reviewer:** challenge uncontrolled capacity/pretraining differences, misuse of tiny-image rate targets, hidden original-image supervision, missing original-training baselines for the dictionary architecture, and claims stronger than the data support. A favorable dictionary result is a hypothesis, not a review requirement.

**Completion evidence for this task:** the current project handoff and course template were read, a proposed design was written, and no dataset/model was selected or run. The proposal, bibliography, and historical project memory were not edited.
