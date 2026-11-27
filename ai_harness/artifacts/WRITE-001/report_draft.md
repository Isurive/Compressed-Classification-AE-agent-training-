# DRAFT: Training an image classifier on heavily JPEG-compressed images: how much storage can we save, and does a sparse dictionary help?

> **Draft for the team, written by the AI agent `claude-orient-20261122-a` (task WRITE-001, 2026-11-26; revision 1 after [REVIEW-012](../REVIEW-012/report.md); revision 2, task WRITE-002, after REVIEW-012's cross-review; revision 3, task COORD-005, 2026-11-27, adds the weight-decay check WD-001 as section 7.4; revision 4, task COORD-006, wording fixes after REVIEW-015).**
> - **Sources:** every result comes from accepted, independently reviewed artifacts. The authoritative list is [`ai_harness/artifacts/COORD-004/status.md`](../COORD-004/status.md); per-task reports are linked in the text. Hardware and timing figures come from `WORK_SUMMARY.md` and the pilot reports; parameter counts and method details come from `src/`.
> - **Layout:** sections 1–6 follow the order of `AI360_Project_Proposal_Template.tex`. Sections 7–10 (results, discussion, limitations, conclusion) are for the final report. The template limits the proposal to 4 pages, so the proposal version must be cut down from this.
> - **Citations:** keys like `[nayak2023deep]` are BibTeX keys. Keys marked † are not in `references.bib` yet; proposed entries are in [`proposed_references.bib`](proposed_references.bib) and must be verified before use.
> - **⚠ TEAM DECISION** boxes mark choices only the team can make. The most important is the **accuracy tolerance** (section 6.4); several sentences change with it.
> - The course `.tex`, `references.bib`, `projectIdea.md` and `src/` were not edited.

---

## 1. Group composition

⚠ **TEAM:** names and student IDs.

| Team member | Student ID |
| --- | --- |
| Member 1 | |
| Member 2 | |
| Member 3 | |

## 2. Introduction and problem definition

Image datasets are expensive to store and move. The 45,000 ImageNet images we train on take 6.04 GB as distributed; full ImageNet is far larger. JPEG can shrink each image many times over, but heavy compression removes detail a classifier might need. We ask a practical question: **if a classifier head on top of a frozen pretrained network is trained only on heavily JPEG-compressed copies of its training images, how much accuracy does it lose on the original, not re-compressed test images, and how much storage does that buy?**

This is a deep-learning problem because the answer depends on what a learned representation is sensitive to. Deep classifiers might tolerate the specific artifacts JPEG introduces, or they might not, and different heads on the same features might tolerate them differently. As a second question, we test whether a **neural sparse-dictionary head** copes with compressed training data better than a conventional linear classifier. Such a head re-expresses each image's features as a sparse combination of learned "atoms" (k of M active; here 512 of 2048).

**Research questions.**
1. How much ImageNetV2 top-1 accuracy is lost when a classifier head is trained on JPEG-compressed copies of its training images (about 50:1 and 86:1 against raw pixels), compared with training on the original files?
2. Does a learned sparse-dictionary head reach higher accuracy, or lose less under compression, than a conventional linear head or a dense (non-sparse) head of the same width?

**Scope.**
- We train only the classification head on top of a frozen ResNet-18 pretrained on ImageNet [he2016deep†]. Our storage claim therefore covers the data used to train the head, not learning a full model from compressed data alone.
- Training images are compressed; all claims use the original ImageNetV2 test images, so the comparison measures training-storage savings rather than robustness to compressed inputs. A secondary score on ImageNetV2 compressed to the training level was recorded but is not used.

## 3. Related work and background

⚠ **TEAM:** the compressed-training part of this section describes only what our literature audit actually inspected ([LIT-001](../LIT-001/report.md)). Four entries already in `references.bib` (`friedland2020impact`, `kuchnik2021progressive`, `pistono2020training`, `yang2024probing`) were **not** inspected in that audit; read and summarize them before citing them for specific claims.

- **Training on compressed images to save storage is established.**
  - Nayak and Friedland [nayak2023deep] fine-tune pretrained classifiers on JPEG-compressed variants of their data, vary model size (EfficientNet B0–B4) and measure storage on disk. So compressed training, model-size comparisons and measured bytes are not individually new.
  - Lee et al. [lee2026storage] study the trade-off between training-data quantity, JPEG quality and storage for classification. In the preprint version we inspected (arXiv v1, §3.1), they test on original images and approximate storage as sample count times JPEG quality factor rather than measuring bytes. We did not inspect the final journal version's methods.
  - Maudgalya [maudgalya2021learned] trains CNNs from scratch on JPEG- and learned-codec-compressed CIFAR-10 and Tiny ImageNet and, in its main experiment, evaluates on original test images. Its rate accounting excludes JPEG headers and coding tables (§3.2, p. 11), so its low bits-per-pixel figures do not translate directly into complete-file savings.
- **Dictionary learning in neural networks** (background; Tang et al. abstract only; Makhzani and Frey not inspected). Dictionary-learning and sparse-coding layers have been combined with deep networks for image recognition, e.g. the deep dictionary learning and coding network of Tang et al. [tang2020dictionary†] (abstract only inspected). Our head is a top-k sparse autoencoder on CNN features, in the spirit of k-sparse autoencoders [makhzani2014ksparse†].
- **Closest unresolved overlap.** Wei et al. study lossy compression and remote-sensing classification with convolutional sparse coding (DOI 10.1109/LGRS.2020.3047789; venue presumably IEEE GRSL, inferred from the DOI and to be verified). We saw only a third-party reproduction of the abstract, not the paper, so we make no claim relative to it.

**How our work relates.** We build on the compressed-training setting of [nayak2023deep, lee2026storage, maudgalya2021learned]. Within that setting, our design uses:
- complete-file byte accounting with headers counted (unlike Lee et al.'s quality-factor proxy and Maudgalya's payload-only rate; Nayak and Friedland also measure on-disk storage);
- a separately collected test set used as distributed (the original ImageNetV2 JPEG files [recht2019imagenet†], never re-compressed by us);
- seed-paired comparisons, with the design and decision rule recorded before the runs;
- a controlled test of whether an explicit sparse-dictionary head changes the trade-off.

The last item is the incremental question; the others are methodological choices, not claimed firsts. We do not claim novelty from the absence of an exact match in our search.

## 4. Datasets

| Role | Dataset | Size | Use |
| --- | --- | --- | --- |
| Training / validation pool | ImageNet-1K (ILSVRC-2012) validation set [russakovsky2015imagenet†], via Hugging Face | 50,000 images, 1,000 classes | Split per seed into 45,000 training and 5,000 validation images (45 and 5 per class) |
| Test | ImageNetV2 matched-frequency [recht2019imagenet†] | 10,000 images (10 per class) | Test only; never used for tuning or checkpoint selection |

- **Why these datasets.**
  - The ImageNet validation set gives 45 labelled training images per class for 1,000 classes, in a size our hardware can process.
  - ImageNetV2 is a separately collected test set with the same classes. A full comparison of file hashes found no byte-identical copy of any of its 10,000 images in our training pool (REVIEW-004; GRID-001); near-duplicates were not checked.
- **Compression.**
  - Each training-pool image was re-encoded at its native resolution with Pillow JPEG (4:2:0 chroma subsampling) at one of several levels. The quality is `round(95 − 94 × level / 100)`, with `optimize=True` (`src/compress_jpeg.py`).
  - The two main levels are **level 80 (quality 20)** and **level 90 (quality 10)**. "Source" means the original ImageNet files, which are themselves JPEGs.
  - Validation images are compressed at the same level as the training images. The test images used for every claim are the original ImageNetV2 files.
- **Preprocessing.** After decoding, standard ResNet-18 ImageNet preprocessing (torchvision `IMAGENET1K_V1` transforms: resize, centre crop to 224 px, normalize). No augmentation, because the backbone is frozen and features are cached.
- **Access.** ImageNet is gated and licensed for non-commercial research only. Each team member must accept its terms; the images are not redistributed. The data live locally in `Datasets/` (see `Datasets/DATASETS.md`).
- **Known issue.** The pool contains 29 groups of byte-identical images with conflicting labels. 3, 3 and 7 groups span the train/validation split for seeds 42, 43 and 44 (not counted for the confirmatory seeds 45–51). This affects only validation-based checkpoint selection, not the ImageNetV2 test.

## 5. Available hardware and computational resources

One laptop: NVIDIA RTX 3080 Ti Laptop GPU (16 GB), Intel i9-12900H, 32 GB RAM. Environment: Python 3.12, PyTorch 2.11 (CUDA 12.8), torchvision 0.26.

Because the backbone is frozen, features are extracted once per compression level and cached. This took about 31 minutes for 8 levels × 60,000 images, about 260 images/s, limited by CPU-bound JPEG encoding, decoding and resizing. One head-training run then takes roughly 30–40 seconds, so an 84-run experiment takes under an hour.

## 6. Model and methodology

### 6.1 Model and starting point

- **Backbone:** ImageNet-pretrained ResNet-18 [he2016deep†], frozen. Each image becomes a 512-dimensional pooled feature vector.
- **Heads**, trained from random initialization (Xavier [glorot2010understanding†] for the classifier, He [he2015delving†] for the encoder) on features of compressed images:

| Head | Architecture (deployed) | Parameters (from layer shapes) |
| --- | --- | --- |
| Linear | 512 → 1,000 | 0.51 M |
| Sparse dictionary, M = 2048, k = 512 | subtract a learned centre, encoder 512 → 2048, keep the top 512 activations (then ReLU), classifier 2048 → 1,000 | 3.10 M |
| Dense control, M = 2048 | the same, but all 2048 activations kept (k = M, ReLU kept) | 3.10 M |
| Sparse dictionary, M = 4096, k = 1024 | the same with encoder 512 → 4096, top 1024, classifier 4096 → 1,000 | 6.20 M |

- **The dictionary.**
  - During training, a decoder with M unit-norm **atoms** (the dictionary) reconstructs the 512-d feature from the sparse code. It adds 1.05 M (M = 2048) or 2.10 M (M = 4096) training-only parameters and is not used at inference.
  - M is the number of available atoms; k is the number active per image.
- **Starting point:** the user-written code in `src/` (`resnet18_sparse_ae.py`, `sparse_workflow.py`, `main.py`). Experiment runners were added as new files and the original files were not modified.

### 6.2 Training objective and optimization

- **Linear head:** cross-entropy.
- **Dictionary heads:** cross-entropy plus feature-reconstruction mean squared error (weight 1).
  - The reconstruction target is the feature of the same **compressed** training image, so no information from the original images enters training.
  - Two extra regularizers in the original code (dependency, coherence) had no measurable effect in our ablation ([ABL-001](../ABL-001/report.md)) and are off in the final experiments.
- **Optimizer:** Adam [kingma2015adam†], batch 128, constant learning rate. Final settings: **lr 1e-4 for the dictionary heads and 3e-4 for the linear head (weight decay 1e-4), 20 epochs for every head.**
- **Where these settings come from:**
  - **Linear head:** 3e-4 for 20 epochs is the setting adopted by the pre-registered, validation-only check [LR-002](../LR-002/report.md).
  - **Dictionary heads:** LR-002's rule adopted 10 epochs for them: sparse M2048 at 3e-4, and M4096 and dense at 1e-4 (following [LR-001](../LR-001/report.md)).
  - **The final 20-epoch, lr 1e-4 setting** is a post-hoc equal-budget choice, disclosed before the confirmatory test. It departs from LR-002 in epochs for all three dictionary heads, and in lr for sparse M2048. On validation it was never worse: +0.28 (sparse M2048), +0.02 (M4096) and +0.11 pp (dense).
- **Checkpoint selection:** the epoch with the best validation top-1 (ties broken by lower validation cross-entropy).
- **Chosen earlier, at lr 1e-3, and not re-tuned:**
  - k (512 tuned; 1024 follows a k = M/4 rule);
  - initialization;
  - the reconstruction weight (1);
  - weight decay (1e-4 for linear, 0 for the dictionary heads).
- **Follow-up check ([WD-001](../WD-001/report.md), section 7.4):** the two 2048-wide heads were also trained with cross-entropy only (reconstruction weight 1e-6, effectively off, as in ABL-001's cross-entropy-only setting) and with tuned weight decay instead. Weight decay was coupled L2 through Adam, on the encoder and classifier weights and biases. Learning rate, weight decay and budget were chosen on validation from 3 lr × 4 weight decays × {10, 20} epochs: lr 1e-4 (dense) and 3e-5 (sparse), weight decay 1e-3, 20 epochs for both.

### 6.3 Baselines and evaluation

- **Baselines.** The linear head is the conventional baseline. The dense head is the same width with sparsity removed, which isolates the effect of top-k sparsity. As a reference, the untouched original ImageNet classifier scores 57.28% top-1 on ImageNetV2.
- **Metrics.**
  - ImageNetV2 top-1 accuracy (top-5 also recorded).
  - Storage: complete-file JPEG bytes with headers counted, bits per pixel, the compression ratio against raw pixels (each image's actual channel count), and the saving against the stored original files.
- **Statistics.**
  - Each seed sets the train/validation split, initialization and batch order.
  - Comparisons are paired by seed: mean difference, paired-t 95% confidence interval and sign count.
  - A difference is **detected** if its CI excludes 0 and **meaningful** if, in addition, its mean is at least the tolerance.
  - Holm correction [holm1979simple†] is applied across each pre-registered test family.
- **Pre-registration.** For the main experiments, the design and decision rule were recorded in a task ledger before the first run (SEED-001, MSWEEP-001, LR-001, LR-002, RETEST-001, WD-001); for SEED-001, MSWEEP-001, RETEST-001 and WD-001 this included the test family. The confirmatory experiments, RETEST-001 and WD-001, used seeds 45–51, which were never used to choose learning rates or weight decay. WD-001's validation choice was locked in the ledger before any of its ImageNetV2 runs.
- **Independent review.** Each experiment was independently reviewed, with numbers recomputed from logs, before acceptance. Section 8 describes how this changed our conclusions.

### 6.4 Accuracy tolerance

> ⚠ **TEAM DECISION: accuracy tolerance.**
> - All verdicts below use **1.0 percentage point (pp)** as the working tolerance. It came from our approved experiment plan ([REASON-002](../REASON-002/plan.md)) but was never formally finalized.
> - **If you keep 1.0 pp,** the text below stands as written.
> - **If you choose 0.5 pp:**
>   - Under the pre-registered rule, six Holm-detected results become meaningful findings:
>     - the 4096-atom head's advantage over linear (+0.65) and over the 2048-atom head (+0.63);
>     - all four level-80 costs: sparse M2048 0.85, sparse M4096 0.79, dense 0.69, linear 0.53.
>   - Only the sparse M2048 and M4096 level-80 costs have CIs entirely above 0.5; the others' CIs dip below it.
>   - The exploratory finding that the dictionary heads lose 0.5–0.7 pp more than linear at level 90 would also reach the tolerance.
>   - The level-90 costs and the sparse − linear and sparse − dense results are unchanged.
>   - The weight-decay results (section 7.4) are unchanged: the dense gaps (−0.29, −0.35) are below 0.5 pp and the sparse gaps (−1.36, −1.34) are above 1.0 pp.
>   - Rewrite sections 7.2, 7.3, 8 and 10, including the "under 1 pp" wording for level 80, which becomes "a meaningful loss for every head".

## 7. Results

All numbers are ImageNetV2 top-1 accuracy from the confirmatory experiment [RETEST-001](../RETEST-001/report.md) (7 seeds, 84 runs, settings of section 6.2), unless stated otherwise.

### 7.1 Storage

Complete-file bytes of the 45,000 training images, encoded at native resolution ([GRID-001](../GRID-001/report.md) §4, seed 42):

| Level (JPEG quality) | Size | Bits per pixel | vs raw pixels | Smaller than original files by |
| --- | --- | --- | --- | --- |
| Source (original files) | 6.04 GB | 4.63 | 5.1 : 1 | — |
| 80 (q20) | 0.616 GB | 0.47 | **50.3 : 1** | **89.8%** |
| 90 (q10) | 0.360 GB | 0.28 | 86.0 : 1 | 94.0% |
| 95 (q6) | 0.238 GB | 0.18 | 130.0 : 1 | 96.1% |
| 100 (q1) | 0.131 GB | 0.10 | 235.8 : 1 | 97.8% |

- Header bytes are included in these sizes (about 14 MB of the 616 MB at level 80).
- The saving against the original files includes metadata stripped on re-encoding.
- The stored files decode to exactly the pixels we trained on (3,500 of 3,500 checked).

⚠ **TEAM:** a storage-vs-accuracy figure would summarize sections 7.1 and 7.2 well. It needs matplotlib, which is not installed yet.

### 7.2 Accuracy cost of compressed training data

Mean accuracy (%), 7 seeds:

| Head | Source | Level 80 | Level 90 |
| --- | --- | --- | --- |
| Linear | 48.94 | 48.42 | 47.43 |
| Dense M2048 | 49.10 | 48.42 | 47.06 |
| Sparse M2048 / k512 | 49.29 | 48.45 | 47.10 |
| Sparse M4096 / k1024 | **49.90** | **49.11** | **47.72** |

Accuracy lost relative to the same head trained on the original files (pp, mean [95% CI], paired by seed):

| Head | Level 80 (50:1) | Level 90 (86:1) |
| --- | --- | --- |
| Linear | 0.53 [0.16, 0.89] | 1.51 [1.06, 1.95] |
| Dense M2048 | 0.69 [0.41, 0.96] | 2.04 [1.60, 2.48] |
| Sparse M2048 | 0.85 [0.52, 1.17] | 2.19 [1.91, 2.46] |
| Sparse M4096 | 0.79 [0.50, 1.08] | 2.18 [1.81, 2.54] |

- **Level 80** costs every head a detected amount that is **under 1 pp on average** (0.53–0.85 pp). This is not proof of "no meaningful loss": the confidence intervals stay below 1 pp for the linear and dense heads but reach 1.08–1.17 pp for the sparse heads.
- **Level 90** costs **1.5–2.2 pp**, a meaningful loss for every head.
- Relative to each head's accuracy when trained on the original files, level 80 retains 98.3–98.9%.
- **Stronger compression** costs much more: 2.7–5.0 pp at level 95 and 7.7–18.7 pp at level 100, across the heads tested then. These were measured in earlier experiments at the older learning rate and not re-tested.

### 7.3 Does the sparse dictionary help?

Pre-registered comparisons, pooled over the three training levels (pp):

| Comparison | Mean | 95% CI | Seeds positive | Holm | Verdict (1 pp tolerance) |
| --- | --- | --- | --- | --- | --- |
| Sparse M2048 − linear | +0.02 | [−0.25, +0.29] | 4/7 | not detected | no detected difference |
| Sparse M2048 − dense M2048 | +0.09 | [+0.002, +0.17] | 5/7 | not detected | small, not robust to multiplicity |
| Sparse M4096 − sparse M2048 | +0.63 | [+0.47, +0.80] | 7/7 | detected | robust, below 1 pp |
| Sparse M4096 − linear | +0.65 | [+0.45, +0.85] | 7/7 | detected | robust, below 1 pp |

- **The sparse-dictionary head (M2048/k512) is no better than a linear classifier** (+0.02 pp, CI [−0.25, +0.29]: no detected difference; this is not an equivalence test).
- **Sparsity at equal width:** sparse minus dense is +0.09 pp on average (CI [+0.002, +0.17]) and not robust to multiplicity. We find no practically relevant effect of top-k sparsity (k = 512 of M = 2048, with the MSE loss).
- **The 4096-atom head has the highest mean at every level.** Pooled over levels it leads the 2048-atom head by 0.63 pp and the linear head by 0.65 pp (7/7 seeds). Its lead over linear shrinks from +0.96 pp at source to +0.29 pp at level 90, where the CI includes 0. It doubles the available atoms, the active atoms and the parameter count at once, so we cannot say which causes the gain; extra capacity is the simplest explanation (an inference).
- **No head reduces the cost of compression** (exploratory). At level 90 the sparse heads lose 0.68 and 0.67 pp more than the linear head (detected, uncorrected; dense +0.53), so the dictionary heads' advantage shrinks toward level 90. This should not be read as the linear head being more compression-tolerant: its smaller absolute cost goes with lower accuracy on the original images.

### 7.4 Is the reconstruction loss needed? Weight decay instead

The dictionary heads are trained with a feature-reconstruction loss next to cross-entropy. An early ablation at the old learning rate found that a 2048-wide head trained with cross-entropy alone did worse than the linear head ([ABL-001](../ABL-001/report.md)). That left an open question: does the reconstruction loss do something specific, or does it just act as a regularizer that tuned weight decay could replace? A pre-registered follow-up, [WD-001](../WD-001/report.md), trained both 2048-wide heads with cross-entropy and tuned weight decay instead (section 6.2), on the same seeds, levels and splits as section 7.2.

Pre-registered comparisons, pooled over the three training levels (pp; Holm over these 4 tests):

| Comparison | Mean | 95% CI | Seeds positive | Holm | Verdict (1 pp tolerance) |
| --- | --- | --- | --- | --- | --- |
| Dense, weight decay − dense, reconstruction loss | −0.29 | [−0.41, −0.17] | 0/7 | detected | below tolerance |
| Sparse, weight decay − sparse, reconstruction loss | −1.36 | [−1.52, −1.20] | 0/7 | detected | **meaningful** |
| Dense, weight decay − linear | −0.35 | [−0.57, −0.14] | 0/7 | detected | below tolerance |
| Sparse, weight decay − linear | −1.34 | [−1.63, −1.05] | 0/7 | detected | **meaningful** |

Mean accuracy (source / level 80 / level 90, %): dense with weight decay 48.90 / 48.17 / 46.66; sparse with weight decay 47.95 / 47.10 / 45.70.

- **Dense head:** tuned weight decay comes close to the reconstruction loss but not all the way (−0.29 pp, detected, below the tolerance). This outcome was not one of the two readings we declared in advance, so this description is post-hoc.
- **Sparse head:** within the tested grid and the 20-epoch budget, the reconstruction loss gives a benefit that tuned weight decay does not (−1.36 pp). The gap is about the same at every level (−1.34 to −1.40 pp). A caveat: the sparse weight-decay head chose the smallest learning rate in the grid, and 16 of its 21 test runs picked epoch 19 or 20 of 20, so it may be under-trained. A longer budget might narrow the gap; that would be a new experiment.
- **Neither weight-decay head beats the linear head.** So the reconstruction loss matters for the sparse head, but with or without it no 2048-wide head is meaningfully better than linear (section 7.3).
- **Compression cost** (exploratory): replacing the reconstruction loss with weight decay showed no detected change in how much compression costs (all differences within ±0.21 pp, CIs including 0; not an equivalence result).
- **Tuning effort** favoured the weight-decay heads: each chose among 24 settings, while the reconstruction-loss heads had only a learning-rate check. So unequal tuning does not explain why they fall short.

## 8. Discussion: a head advantage that depended on the learning rate

Our earlier experiments used Adam with lr 1e-3 for 10 epochs for every head. At that setting the sparse head beat the linear head by +0.97 pp [0.78, 1.15] on these same seeds, in all 7 seeds ([SEED-001](../SEED-001/report.md), pre-registered). Sparsity appeared to add +0.41 pp over the dense head.

An independent reviewer then noticed that the larger dictionary heads were unstable at lr 1e-3, which led to two validation-only learning-rate checks:
- lr 1e-3 was too high for the dictionary heads: their training loss rose partway through training in 4 to 6 of 6 runs per head.
- More importantly, the **linear baseline was under-tuned by about 2 pp**.

Every head was then re-trained at a validation-chosen learning rate and a common 20-epoch budget (the earlier runs used 10 epochs) and re-tested on ImageNetV2. Every head improved: linear by +1.95 pp and the dictionary heads by +1.0 to +1.3 pp (exploratory; the gain mixes the learning-rate and budget changes). The gap closed because the linear head gained about 0.95 pp more than the sparse head, leaving +0.02 pp.

The earlier measurements were correct as measured; the head comparison drawn from them depended on the learning rate. A practical lesson for comparing architectures is to **tune the baseline as carefully as the proposed model, and confirm small effects with a pre-registered test on held-out seeds.**

The storage result held up through the same process. At lr 1e-3 the costs were about 0.5–1.2 pp at level 80 and 1.5–2.3 pp at level 90; at tuned rates they are 0.53–0.85 pp and 1.51–2.19 pp.

## 9. Limitations

- **Frozen pretrained backbone.** Only the head is trained on compressed images. The backbone learned from original ImageNet images, so we do not show that a model can be learned from compressed data alone. End-to-end fine-tuning is a different and larger claim.
- **One test set, reused.** ImageNetV2 was scored many times during the project, and its results guided which experiment we ran next, although no setting was tuned on it. Confidence intervals cover seed-to-seed variation on one fixed set of 10,000 images. The extra uncertainty from the choice of test images was not quantified; REVIEW-010's rough estimate (an inference, not measured) is about 0.2–0.3 pp per model pair, the same order as the smallest effects above.
- **Confounded dictionary size.** The 4096-atom result changes M, k and the parameter count together, with no dense M4096 or parameter-matched control.
- **Settings not re-tuned at the final learning rate:** k, initialization, the reconstruction weight, and weight decay for the linear head and the reconstruction-loss heads. Weight decay was tuned only for the cross-entropy-only heads of section 7.4, with coupled L2 (not AdamW), at M = 2048 only; the sparse one may be under-trained.
- **Small per-class training set.** With 45 training images per class, results may differ from training on the full ImageNet training set (about 1,280 images per class on average; 1.28 M images).
- **Absolute accuracy and the 98% target.** The proposal's stretch target of 98% absolute accuracy above 50:1 is not met. At level 80 (50.3:1) the heads reach about 48.4–49.1% on ImageNetV2; the original classifier, trained on the full 1.28 M-image ImageNet training set, reaches 57.28%. A different criterion, retention relative to each head's accuracy when trained on the original files, gives 98.3–98.9% at level 80.
- **One codec and one setting:** Pillow 13.0.0 JPEG with 4:2:0 chroma subsampling, encoded at each image's native resolution. The model sees 224-px centre crops of the decoded images. Other codecs, resolutions or encoders were not tested.
- **Storage only.** Smaller files say nothing about GPU memory or training speed. With cached features, training compute is the same at every level.

## 10. Conclusion

- **Storage.** Storing the training images as JPEG quality 20 shrinks them about 50-fold against raw pixels (50.3:1). That is 89.8% smaller than the original files, a saving that includes metadata stripped on re-encoding. A classifier head trained on them, on frozen ImageNet-pretrained ResNet-18 features, loses on average 0.53–0.85 pp of ImageNetV2 accuracy. The 95% CIs stay below 1 pp for the linear and dense heads but reach 1.08–1.17 pp for the sparse heads, so this is not an equivalence result. At quality 10 (86:1) it loses 1.5–2.2 pp.
- **Sparse dictionary.** At properly tuned learning rates it does not beat a linear classifier (no detected difference) and does not make the head more tolerant of compression. Only the M4096/k1024 head is slightly more accurate (+0.6 pp pooled, below our tolerance). It doubles the available atoms, the active atoms and the parameters at once, so the cause is not identified.
- **Reconstruction loss.** For the sparse head it is not replaceable by tuned weight decay within our grid and budget (−1.36 pp without it); for the dense head weight decay comes close but not all the way (−0.29 pp, detected, below our tolerance). Either way, no 2048-wide head beats the linear classifier.
- **Method.** An apparent 1-pp advantage of the sparse head at a shared learning rate (Adam 1e-3, 10 epochs) did not survive validation-chosen learning rates, mainly because the linear baseline had been under-tuned. A pre-registered re-test on ImageNetV2 confirmed this.

⚠ **TEAM:** adjust the conclusion wording if the tolerance changes (section 6.4).

---

## Appendix A. Timeline and division of work

⚠ **TEAM:** the template asks that every table be completed. These are team facts; the agent has not filled them in. For reference, the experiment record by date is in [`WORK_SUMMARY.md`](../../../WORK_SUMMARY.md) section 2, and the coordination harness and review process are in section 6.

| Time period | Milestone / activities | Expected deliverable |
| --- | --- | --- |
| | | |

| Team member | Primary responsibilities | Expected contributions |
| --- | --- | --- |
| | | |

## Appendix B. Reproducibility

- **Code:** `src/` (the user's model code, unchanged) and the runners `run_grid.py`, `run_grid_xavier.py`, `run_ablation.py`, `run_seeds.py`, `run_msweep.py`, `run_lrcheck.py`, `run_lrcheck2.py`, `run_retest.py`, `run_wdcheck.py`.
- **Commands and runtimes:** `WORK_SUMMARY.md` section 4.
- **Records:** every experiment's pre-registration (`task.json`), analysis script, logs and independent review are in `ai_harness/artifacts/<TASK>/`.
- **Results tables:** `runs/<TASK>/results.csv`.
- **Public repository:** <https://github.com/Isurive/Compressed-Classification-AE-agent-training->. Datasets and checkpoints are not included.

## Revision log

- **Revision 1 (2026-11-26), for [REVIEW-012](../REVIEW-012/report.md)** (verdict revise, wording only; no number changed):
  - **S-1:** frozen-backbone scope added to the question; storage conclusion gives the CI qualifier and the metadata note.
  - **S-2:** where the learning rates came from (section 6.2) is corrected: 20 epochs for the dictionary heads is a post-hoc equal-budget choice for all three.
  - **S-3:** the M4096 margin is given per level (+0.29 vs linear at level 90); "larger dictionary" wording removed from the conclusion.
  - **S-4:** qualifiers restored for sparse − linear and sparse − dense.
  - **S-5:** the discussion is retitled, notes the 10→20-epoch change and that the gap closed because linear gained more; the conclusion's "Method" bullet is reworded.
  - **S-6:** "How our work relates" lists methodological choices, not additions over prior work.
  - **S-7:** the 0.5 pp tolerance box is corrected (six results; CI caveats; level-90 extra cost).
  - **S-8:** codec limitation corrected (native-resolution encoding, 4:2:0, 224-px crops after decoding).
  - **S-9:** the 98% target is stated as not met; retention given as a different criterion.
  - **O-1 to O-20** applied, except O-11: Wei et al. stays a DOI-only mention, with no bib entry, since only a third-party abstract was seen.
- **Revision 2 (2026-11-26, task WRITE-002)**, for the cross-review of REVIEW-012 by `reviewer-20261126-p` (wording only; no number changed):
  - ImageNetV2 is no longer called "never-compressed" or "uncompressed": its files are JPEGs, used as distributed and never re-compressed by us. This error came from REVIEW-012's suggested S-6 wording.
  - The duplicate-group note gives GRID-001's counts (3, 3, 7 for seeds 42–44).
  - Section 3 says Makhzani and Frey was not inspected, and marks the Wei et al. venue as inferred from the DOI.
  - Maudgalya evaluates on original test images "in its main experiment".
  - The image-sampling estimate is "per model pair".
  - "Uncompressed accuracy" is now "accuracy when trained on the original files".
- **Revision 3 (2026-11-27, task [COORD-005](../COORD-005/task.json); user: "Yes, update them")**, after the weight-decay check WD-001 was accepted (REVIEW-013, REVIEW-014, both cross-reviewed). Numbers from WD-001's `analysis.md`; wording follows REVIEW-014 and its cross-review:
  - new **section 7.4** (weight decay instead of the reconstruction loss);
  - section 6.2: a follow-up-check bullet; section 6.3: WD-001 added to the pre-registered and confirmatory experiments;
  - section 6.4: the tolerance box notes the WD-001 verdicts do not change at 0.5 pp;
  - section 9: the "not re-tuned" limitation now says where weight decay was tuned;
  - section 10: a "Reconstruction loss" bullet; Appendix B: `run_wdcheck.py`.
  - No earlier number or verdict changed.
  - Accepted after [REVIEW-015](../REVIEW-015/report.md), cross-reviewed by `reviewer-20261127-w`.
- **Revision 4 (2026-11-27, task [COORD-006](../COORD-006/task.json))**, wording only, no number changed, for REVIEW-015:
  - **O-2:** section 10's dense clause says "close but not all the way (−0.29 pp, detected, below our tolerance)";
  - **O-4:** section 6.3 says seeds 45–51 were never used to choose "learning rates or weight decay". Revision 3 had widened this to "settings"; seeds 45–51 were also MSWEEP-001's test seeds, and its results informed which heads RETEST-001 included;
  - **O-7:** section 6.2's follow-up bullet gives the cross-entropy-only reconstruction weight (1e-6, effectively off).
