# Project idea and conversation handoff

This document exports the relevant context of the project-planning conversation for another agent or collaborator. It records the user's intentions, the advice discussed, the literature found, and unresolved decisions. **Suggestions are not final decisions unless explicitly identified as user requirements.** No model has been implemented or trained in this session.

## 1. Read this first: the current direction

The user is planning an **AI 360 deep-learning course research project with three people and approximately one month to complete it**. The project must involve a neural network. The original topic of interest is **dictionary learning**.

The latest intended contribution is to investigate whether **heavily compressed training images can replace higher-quality training images while maintaining classification performance and reducing the storage required for the training dataset**.

The user clarified their motivation in their own words:

> "first I want to say its more proving that extremely compressed image can still be classified by using a neural network model, therefore we can reduce the training dataset size in the future by using extremely compressed images instead of the image dataset we currently use."

They also asked whether they must code their own backbone and learning functions, or can adapt an existing open-source model. The recommendation was to **reuse an existing architecture and standard framework components, adapting the dictionary-learning component and experimental pipeline**. Their course proposal template explicitly includes pretrained models, fine-tuning, training from scratch, and adapting or combining existing architectures as possible starting points.

An important interpretation discussed: "reduce dataset size" should mean **reduce storage in bytes while retaining the same training examples**, rather than reduce the number of examples. The user has not yet selected a dataset, exact architecture, or final experiment design.

## 2. User requirements and preferences

- Team: three people.
- Time: approximately one month. The template's illustrative five-week schedule does not override this constraint.
- Must involve a neural network; a purely classical dictionary-learning project is insufficient.
- Wants a research contribution that is not simply an exact repetition of an existing project.
- Explicitly accepts a modest extension of prior research; a fundamentally new algorithm is not required by their stated preference.
- Has repeatedly asked whether proposed ideas have already been done. Be candid about overlap and uncertainty; do not guarantee novelty from an unsuccessful search.
- Is exploring and learning the concepts. Explain terminology and implementation choices plainly.
- Latest emphasis: training-dataset storage savings, rather than only restoration quality or robustness at inference.
- The user asked for this document so other agents can understand the full context and collaborate. This does not itself request creation of another chat or spawning an agent.

## 3. How the idea evolved

1. The user initially asked what dictionary learning is and how it differs from attention.
2. They wanted an original or incrementally novel dictionary-learning project suitable for three people in a month, with a neural-network component.
3. They considered reconstructing compressed files/data, then focused on images.
4. They asked how the number of available dictionary atoms affects reconstruction at different compression levels.
5. They explored JPEG versus PNG and other formats, and how to define compression strength.
6. They shifted to **how dictionary size affects image classification accuracy at different JPEG compression levels**, hoping classification would be easier than reconstruction.
7. They proposed achieving **98% classification accuracy at compression ratios above 50:1**.
8. They clarified that the main motivation is **replacing higher-quality training images with heavily compressed training images to reduce storage**, and asked about reusing open-source models.

Earlier ideas involving rare visual details, decimal points, or diagram connectivity were brainstorming directions, not adopted requirements. Do not bring them back as if they were the current project.

## 4. Core concepts already explained

### Dictionary size and sparsity are different

Dictionary learning represents a patch or feature vector approximately as:

`x ≈ D a`

- `D` contains learned dictionary atoms: reusable patterns or basis vectors.
- `M` is the number of **available** atoms in the dictionary.
- `a` contains coefficients describing how atoms combine to represent the input.
- `k` can denote the number of **active** atoms, for example under a top-k sparsity rule.

Varying `M` while fixing `k` is a possible controlled experiment. An L1 penalty does not necessarily fix the exact active count. In convolutional sparse coding, atoms are filters and coefficients are spatial maps, so any definition of an active-atom budget must specify where it applies.

Changing the width of an ordinary CNN does not automatically constitute dictionary learning. A dictionary component should have an explicit representation role, such as sparse coefficients tied to a learned reconstruction dictionary or an unrolled sparse-coding procedure.

More compression does not necessarily require more atoms. Additional capacity may help, have diminishing returns, or fail to recover information that compression has removed. These are hypotheses to investigate, not established outcomes.

Train separate dictionary sizes for the simplest comparison. Arbitrarily truncating an already-trained dictionary is a different experiment and was not recommended as the default.

### Compression and file formats

- Intact lossless compression such as PNG can be decoded exactly. Changing PNG compression effort generally changes storage/time without changing the decoded pixel values for the same representation.
- Lossy JPEG removes information. Neural restoration can estimate plausible detail, but cannot guarantee recovery of the original image.
- Converting an already damaged JPEG to PNG does not restore lost detail.
- JPEG, lossy WebP, and optionally AVIF were discussed, but **JPEG is the current focus**. Multiple codecs would enlarge the scope and are not required.
- The proposed classifier ordinarily receives **decoded image pixels**, not raw JPEG bytes or compressed-domain coefficients. Compressed-domain classification would be a different, more involved design choice.

### Defining compression strength

Use actual encoded sizes, with dimensions and pixel format fixed:

`bpp = 8 × compressed_file_bytes / (width × height)`

For 8-bit RGB:

`raw_pixel_bytes = 3 × width × height`

`R_raw = raw_pixel_bytes / compressed_file_bytes = 24 / bpp`

Thus 50:1 relative to raw RGB corresponds to 0.48 bpp and a compressed file containing 2% of the raw byte count. For 8-bit grayscale the uncompressed rate is 8 bpp, not 24.

JPEG quality factor 50 is not a 50:1 compression ratio. The same quality factor produces different file sizes for different images, and quality-factor values are not directly comparable across codecs.

For a storage-saving claim, also measure:

`R_existing = total_bytes_in_existing_dataset / total_bytes_in_compressed_dataset`

The source dataset may already contain JPEGs. A 50:1 ratio relative to raw pixels does not imply 50-fold savings over its existing files. Report the reference used for every ratio. Keep the same examples and resolution; do not inflate the baseline by upsampling images merely to obtain a larger apparent ratio.

File-header overhead can make very high full-file compression ratios impractical for tiny images. Check achievable rates in a pilot and report actual rates, including how unattainable targets are handled. Do not assume a quality setting delivers the requested rate for every image.

## 5. The crucial experimental change for the latest goal

**Accurate inference on compressed test images alone does not demonstrate that the training dataset can be stored more aggressively.** The main experiment must train on compressed versions of the training images.

Suggested structure, not yet finalized:

| Condition | Training images | Evaluation images | Purpose |
| --- | --- | --- | --- |
| Reference | Dataset as supplied | Fixed, held-out original test set | Establish reference accuracy |
| Compressed training | Compressed versions of the same training examples | The same original test set | Test whether compressed training data can replace the originals |
| Additional evaluation | Models from the above conditions | Compressed versions of the held-out test set | Assess compressed deployment inputs |

If comparing a conventional classifier against a dictionary-based classifier, train both under comparable data conditions. An original-data condition for each architecture helps distinguish clean-data capacity from the effect of compression.

Controls and measurements discussed:

- Split by original image identity before creating compressed variants; variants of an image must not leak across training and test splits.
- Hold sample count, labels, spatial resolution, and training budget constant when isolating compression effects.
- Control or disclose parameter count, model depth, sparsity settings, pretraining, and tuning budget.
- Use validation data to choose hyperparameters, accuracy tolerances, and compression settings. Reserve the test set for final evaluation.
- Measure absolute test accuracy, accuracy change relative to the relevant original-training baseline, actual stored bytes, and model size.
- Repeat with multiple random seeds where feasible. Three seeds was a suggestion, not a requirement already agreed with the user.
- Store actual compressed training files or byte streams when claiming practical storage reduction. Generating a compressed view from originals each epoch is useful augmentation but does not by itself demonstrate a smaller stored dataset.
- A compressed-only condition should not require original images as reconstruction targets or other hidden training inputs.
- Smaller files do not automatically yield lower GPU memory use or faster neural-network computation. Decoding at unchanged dimensions usually produces tensors of the same shape. Storage and transfer savings are the primary claim; timing and memory benefits require measurement.

## 6. The 98% accuracy target

The user proposed reaching **98% absolute accuracy at ratios above 50:1**. This has not been abandoned, but it was recommended as a **stretch target subject to a dataset-specific feasibility pilot**.

Reasons discussed:

- Feasibility depends on dataset difficulty, number of classes, resolution, model, pretraining, and reference accuracy.
- If reference accuracy is below 98%, the target needs more justification.
- If a conventional baseline already reaches 98% under the same compression, reaching 98% alone is not a new result.
- "Above 50:1" needs a bounded tested range; it cannot mean arbitrary compression without limit.
- Empirical results on one benchmark support conclusions about that setting, not a proof that all datasets can be aggressively compressed without loss.

Alternative practical criteria discussed were remaining within one or two **percentage points** of a reference accuracy, or selecting the smallest tested dictionary that meets a validation-defined tolerance. These were suggestions, not final user choices. Absolute 98% accuracy, retaining 98% of baseline accuracy, and losing at most two percentage points are different criteria.

The latest project question can be expressed as:

> Does a neural dictionary-based classifier allow training images to be compressed more aggressively than a comparable conventional classifier allows, while maintaining a specified level of held-out classification accuracy?

Dictionary size can be a secondary research variable within this storage-focused question.

## 7. Reusing models versus implementing everything

The recommendation was to **adapt an existing implementation**, rather than spend the month implementing all numerical machinery.

| Component | Suggested approach |
| --- | --- |
| Backbone / feature extractor | Reuse an established small CNN implementation appropriate for the dataset |
| Learned dictionary and sparse coding | Adapt existing research code or implement a focused module with framework primitives |
| Classification head | Standard linear layer |
| Classification loss | Standard cross-entropy implementation |
| Optimizer and gradients | Standard optimizer and automatic differentiation |
| Compression pipeline, controlled comparisons, dictionary-size study, and analysis | Project-specific implementation and research work |

The user asked about coding "regression functions." The task is classification; standard classification losses and framework components are appropriate. A novel optimizer or handwritten backpropagation is not necessary to investigate the research question.

**Existing architecture code is not the same as pretrained weights.** An existing architecture can be trained from random initialization. Torchvision's ResNet-18 is one example supporting both modes; it was mentioned illustratively, not selected as the final model.

For the broad claim of learning from compressed training data, training a small reused architecture from random initialization is a clean main experiment if resources permit. With pretrained weights, the directly supported storage claim concerns the downstream **fine-tuning dataset**, since earlier training already occurred. A pretrained experiment is valid if described accordingly.

A possible neural dictionary pipeline previously discussed was:

`JPEG file → decode → feature extraction / neural sparse encoder → sparse coefficients → classifier`

The dictionary may be tied to an auxiliary reconstruction objective for patches/features, or implemented through unrolled sparse coding. The precise architecture is unresolved. Standard neural dictionary methods exist; merely adding a generic hidden layer is not sufficient to claim dictionary learning.

## 8. Prior work and novelty: verified findings and limits

**Do not describe the broad idea as the first demonstration of classifying or training on compressed images. Existing research already does this.** The original narrow dictionary-size-by-JPEG-compression grid was not located as a verified exact duplicate in the targeted searches, but that is not a novelty guarantee.

### Closest work for the latest training-storage motivation

1. **Nayak and Friedland, "Deep Layers Beware: Unraveling the Surprising Benefits of JPEG Compression for Image Classification Pre-processing" (2023).** Reports comparable accuracy with compressed images occupying as little as 14% of the original dataset size in its experiments. Also discusses I/O benefits and does not report corresponding advantages in convergence, model size, or noise robustness. This directly overlaps with the broad storage-saving motivation.
   - https://www.amazon.science/publications/deep-layers-beware-unraveling-the-surprising-benefits-of-jpeg-compression-for-image-classification-pre-processing
   - Evidence inspected: author-hosted publication summary; search also returned the paper PDF.

2. **"Toward storage-aware learning with compressed data an empirical exploratory study on JPEG" (ICT Express, 2026; DOI 10.1016/j.icte.2025.12.005).** Studies JPEG training-data quantity versus quality under storage budgets using CIFAR-10. This is particularly close to the latest motivation.
   - https://www.sciencedirect.com/science/article/pii/S2405959525002048
   - Evidence inspected: publisher abstract returned in search. Direct full-page access returned 403. Full experimental details were not reviewed.

3. **"Using Learned Image Compression for Training" (Berkeley technical report, 2021).** Examines training classifiers with compressed data, including JPEG and learned compression, on CIFAR-10 and Tiny ImageNet. Search excerpts show training and evaluation comparisons and discussion of storing compressed representations.
   - https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-98.pdf
   - Evidence inspected: relevant indexed excerpts, not a complete methodological review.

4. **Janeiro et al., "Are Visual Recognition Models Robust to Image Compression?" (2023).** Evaluates recognition under compression and reports that fine-tuning on compressed training images recovers much of the lost performance. This supports including a strong compression-trained conventional baseline.
   - https://arxiv.org/abs/2304.04518
   - Evidence inspected: author abstract. Do not attribute segmentation-specific numerical results to image classification.

### Closest work for dictionary capacity, compression, and classification

5. **Wei et al., "Effects of Lossy Compression on Remote Sensing Image Classification Based on Convolutional Sparse Coding" (2021 online / 2022 journal volume).** Uses online convolutional dictionary learning, ADMM sparse coding, and multiple kernel learning to estimate classification accuracy under lossy compression.
   - DOI: https://doi.org/10.1109/LGRS.2020.3047789
   - Accessible abstract: https://www.researchgate.net/publication/348858014_Effects_of_Lossy_Compression_on_Remote_Sensing_Image_Classification_Based_on_Convolutional_Sparse_Coding
   - **Important unresolved source:** full experimental section was not accessible. We cannot rule out dictionary-size experiments inside it. Do not claim it definitely lacks them. Its abstract describes accuracy estimation; do not casually equate that with training the proposed neural content classifier.

6. **Tu et al., "Dictionary learning-based super-resolution restoration for detecting low-quality facial forged videos" (2026).** Uses spatial/temporal dictionaries for compressed-video forgery detection. Table 4 explicitly concerns dictionary-size and patch-resolution sensitivity under heavy compression (FF++, C40). The paper discusses H.264 compression conditions.
   - https://link.springer.com/article/10.1007/s44443-026-00596-8
   - Evidence inspected: full article text, table caption, compression and method descriptions. Exact Table 4 numerical cells were not retrieved.
   - This already connects dictionary size, compression, and a classification task. It differs from the proposed JPEG still-image training-storage study, but prevents a broad "nobody has studied dictionary size and classification under compression" claim.

7. **Fu et al., "JPEG Artifacts Reduction via Deep Convolutional Sparse Coding" (ICCV 2019).** Section 3.3.2 / Table 4 tests 32, 64, and 128 filters and 20, 40, and 60 iterations on LIVE1 at JPEG quality 10. The outputs are restoration metrics, not content-classification accuracy.
   - https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_JPEG_Artifacts_Reduction_via_Deep_Convolutional_Sparse_Coding_ICCV_2019_paper.pdf#page=7
   - Evidence inspected: indexed full section and table. At 40 iterations, the reported PSNR values are 29.14, 29.17, and 29.19 for 32, 64, and 128 filters respectively. These are that paper's results, not project measurements.

8. **Tang et al., "When Dictionary Learning Meets Deep Learning: Deep Dictionary Learning and Coding Network for Image Recognition With Limited Data" (2021).** Establishes neural dictionary-based recognition as prior work.
   - https://arxiv.org/abs/2005.10940
   - Author-hosted PDF: https://robotics.pkusz.edu.cn/static/papers/ITNNLS-tanghao.pdf
   - Reported code repository: https://github.com/Ha0Tang/DDLCN
   - The repository was not successfully inspected or run. Do not assume its environment, license, implementation language, or suitability has been verified.

### Additional related work encountered

- **Mairal, Bach, and Ponce, "Task-Driven Dictionary Learning":** supervised dictionary learning for tasks including classification. https://arxiv.org/abs/1009.5358
- **Qi et al., "End-to-End Optimization of JPEG-Based Deep Learning Process for Image Classification" (2023):** jointly considers JPEG settings and classification under bandwidth constraints. It is not evidence that the proposed exact dictionary-capacity study is new. https://arxiv.org/abs/2308.05840
- **"Effects of JPEG Compression on Vision Transformer Image Classification for Encryption-then-Compression Images" (2023):** reports above-98% accuracy in a setting using CIFAR-10 enlarged from 32×32 to 224×224 and an ImageNet-21k-pretrained ViT. Its numerical results are not directly comparable to the user's proposed 50:1 target. https://pmc.ncbi.nlm.nih.gov/articles/PMC10098741/
- **"Multiple JPEG Compression Detection Through Task-Driven Non-Negative Matrix Factorization" (2018):** predicts how many JPEG compression cycles an image underwent, not its semantic content class. Its experiment uses a 189×57 dictionary. https://nicobonne.github.io/assets/papers/20180216013612_713116_2093.pdf
- **"FA-CDDL: Contrastive deep dictionary learning with frequency augmentation" (2026):** relevant deep dictionary recognition/robustness work. Accessible abstract and introductory material discuss frequency perturbations; the full experimental section was not checked, so do not claim a confirmed JPEG-size sweep or its absence. https://doi.org/10.1016/j.knosys.2026.115771
- **Hendrycks and Dietterich corruption benchmarks:** useful background and possible JPEG evaluation infrastructure. CIFAR-10-C contains existing test corruptions, but using it does not by itself construct the compressed training-data storage experiment. https://github.com/hendrycks/robustness

## 9. Course proposal requirements actually inspected

Workspace:

`C:\Users\ericw\OneDrive\Documents\AI 360 (DeepLearning`

Relevant files:

- `Final Project\AI360_Project_Proposal_Template.tex`
- `Final Project\AI360_Project_Proposal_Template.pdf`

The `.tex` template was read. It asks for a proposal of **at most four pages excluding references**. It requires completing each table and strongly suggests retaining the provided sections.

Sections include:

1. Group composition.
2. Introduction and problem definition.
3. Related work and background.
4. Datasets, splits, preprocessing, and access/computational considerations.
5. Available hardware and computational resources.
6. Proposed model, starting point, methodology, baselines, and evaluation.
7. Timeline and deliverables, including a feasibility milestone.
8. Division of work.

The introduction specifically requests **one paragraph** addressing:

- What problem are you trying to solve?
- What is its motivation and potential significance?
- What makes it suitable for deep learning?
- What specific research question or hypothesis will you investigate?

Lines 127–133 of the template list existing pretrained models, fine-tuning, training from scratch, and adapting/combining architectures as starting points. No additional course rules about code reuse were discovered in this session. Cite and identify reused code and modifications.

## 10. Draft wording discussed, not an approved final proposal

The most recent suggested contribution paragraph was:

> This project investigates whether heavily JPEG-compressed images can replace higher-quality training images while reducing dataset storage requirements and maintaining classification performance. We will adapt an existing neural architecture to incorporate sparse dictionary learning and compare it with a conventional classifier trained on the same compressed data. By varying compression strength and the number of available dictionary atoms, we will examine whether dictionary learning enables greater storage reduction at a comparable level of accuracy. Evaluation on a fixed held-out test set will quantify the relationship between training-data storage and classification performance, identifying the compression conditions under which the proposed approach remains effective.

Possible research question:

> At a fixed number of training examples, can a neural dictionary-based classifier learn from more aggressively JPEG-compressed training images than a comparable conventional classifier while retaining similar test accuracy, and how does dictionary size affect this trade-off?

Possible hypothesis, not an established fact:

> Learned sparse representations will reduce the accuracy loss caused by compressing the training dataset, with diminishing benefits from increasing dictionary size at the strongest compression settings.

A negative or mixed result can still answer the research question. Do not frame success as guaranteed, or describe a small empirical study as a universal proof.

## 11. Open decisions and sensible next steps

The following remain unresolved:

- Dataset and semantic classes; class balance and reference accuracy.
- Native or standardized image resolution and achievable full-file JPEG rates.
- Available GPUs, memory, and total training time.
- Exact neural dictionary architecture and implementation starting point.
- Whether the main experiment uses random initialization or pretrained weights.
- Dictionary sizes, active-atom budget or sparsity penalty, and training objective.
- Final compression targets, whether per-image or dataset-level, and handling of unattainable targets.
- Accuracy tolerance and whether 98% remains a viable stretch target.
- Number of seeds and breadth of comparisons that fit the month.
- How much originality the instructor expects beyond replication and controlled extension.

Previously suggested illustrative values were dictionary sizes 64/128/256 and active budget k=8, but they have not been selected. Earlier JPEG quality levels 90/50/20/10 were convenience controls, not equivalent to 50:1 ratios.

Recommended next work:

1. Pick a manageable dataset and architecture, informed by actual hardware.
2. Check the closest training-storage and dictionary/compression papers, especially the unresolved Wei et al. experiments, before making a novelty claim.
3. Run a small feasibility pilot: achievable JPEG sizes, reference accuracy, compressed-training accuracy, and training runtime.
4. Finalize a controlled comparison that isolates compression and the dictionary component.
5. Draft the course proposal using the template, with modest, testable claims and a realistic month-long schedule.

No dataset was downloaded, no code or notebook was implemented, no training was run, and the proposal template was not edited during the planning conversation. This handoff file is the first project artifact created from that discussion.
