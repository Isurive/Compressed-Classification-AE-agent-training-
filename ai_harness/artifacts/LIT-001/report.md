# LIT-001: first-pass literature audit

Date: 2026-11-05. Role: paper researcher. Status: evidence collection complete for this bounded pass; novelty assessment remains provisional.

Inputs read: `projectIdea.md` and `references.bib`. This report adds evidence without changing those historical records. No dataset was downloaded, no model was trained, and no source repository was executed. Proposed experiments below are recommendations, not decisions made by the user.

## Main finding

The general claim that classifiers can be trained or fine-tuned with JPEG-compressed images while reducing dataset storage is already established research. A generic model-capacity-versus-compression study also overlaps prior experiments. A more defensible incremental project is to measure whether **explicit neural sparse-dictionary capacity, with a fixed active-atom budget, changes the storage/accuracy trade-off** relative to controlled conventional models. This audit does not establish that the exact combination is new.

## Evidence matrix

| Work | Evidence actually inspected | Relevant overlap and limits |
|---|---|---|
| Nayak and Friedland, *Deep Layers Beware* (2023) | Full primary PDF, sections II-A–C and III; [Amazon author page](https://www.amazon.science/publications/deep-layers-beware-unraveling-the-surprising-benefits-of-jpeg-compression-for-image-classification-pre-processing), [working PDF](https://cdn.amazon.science/61/96/9dcc72564f34a8f5652112be022f/deep-layers-beware-unraveling-the-surprising-benefits-of-jpeg-compression-for-image-classification-pre-processing.pdf). | Fine-tunes pretrained classifiers on JPEG variants; varies EfficientNet B0–B4; measures storage on disk. Therefore compressed training, ordinary model-capacity comparisons, and actual bytes are not individually new. Its compression-ratio notation should not be copied as a verified raw-to-JPEG ratio. |
| Lee et al., *Toward storage-aware learning with compressed data: An empirical exploratory study on JPEG* (2026 journal article; 2025 preprint) | Primary [arXiv v1 full text](https://arxiv.org/html/2508.12833v1), sections 3.1–3.3. [Publisher record](https://www.sciencedirect.com/science/article/pii/S2405959525002048) identified; final published methods not inspected. | Directly studies training-data quantity/quality and storage-aware classification. Section 3.1 approximates storage by sample count times JPEG quality factor and tests on original images. These details are verified for the preprint, not assumed unchanged in the final article. |
| Maudgalya, *Using Learned Image Compression for Training Accurate and Robust Convolutional Neural Networks* (Berkeley, 2021) | Primary [technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-98.pdf), sections 3.2–3.3, printed pages 9–17. | Trains from scratch on compressed CIFAR-10/Tiny ImageNet and evaluates original test images in the main experiment; includes JPEG and learned codecs. Also varies CNN width/depth, but that comparison uses matched compressed test data. This is close methodological prior art. |
| Wei et al., *Effects of Lossy Compression on Remote Sensing Image Classification Based on Convolutional Sparse Coding* | [DOI](https://doi.org/10.1109/LGRS.2020.3047789) identified but inaccessible through the browser tool. Only a [third-party reproduction of the abstract](https://www.researchgate.net/publication/348858014_Effects_of_Lossy_Compression_on_Remote_Sensing_Image_Classification_Based_on_Convolutional_Sparse_Coding) was available. | Unresolved high-priority overlap. The reproduced abstract describes dictionary-based sparse features and prediction of classification accuracy. This is a search lead, not sufficient primary-text evidence to establish its codec, capacity sweep, or training-storage protocol. |
| Tang et al., *When Dictionary Learning Meets Deep Learning: Deep Dictionary Learning and Coding Network for Image Recognition with Limited Data* | Primary [arXiv abstract](https://arxiv.org/abs/2005.10940), submitted 2020 and marked accepted to TNNLS. Full methods/code not inspected in this pass. | Establishes dictionary-learning/coding layers for neural image recognition. It is a candidate architecture reference, not evidence of JPEG-storage results or a guaranteed suitable implementation. |

## Two measurement findings that change interpretation

1. The Lee preprint's sample-count-times-quality-factor proxy is not encoded storage. JPEG quality is an encoder setting; the proposed project should use measured bytes. This improves on that particular accounting choice, while Nayak already measures disk storage.
2. Berkeley section 3.2, printed page 11, excludes JPEG header segments and Huffman/quantization-table bytes from its rate accounting. Its reported 0.48 bits per pixel therefore does **not** demonstrate 50:1 storage savings using complete independent JPEG files. Figure 3.4's Tiny ImageNet metric is top-5, so it must not be compared directly with a proposed top-1 accuracy target.

The working Amazon CDN URL upgrades the prior access limitation in `projectIdea.md`: Nayak's methods are now inspected. The accessible Lee author preprint similarly upgrades the earlier abstract-only evidence, subject to the version caveat. Existing files were not modified.

## Recommended falsifiable contribution

Research question: **For a fixed set of training examples, does increasing the number of available atoms M in a neural sparse-coding module, while holding the active budget k fixed, reduce the complete JPEG bytes needed to preserve classification accuracy on original held-out images? Does it outperform an appropriately controlled conventional classifier?**

Suggested operational hypothesis: at least one dictionary setting requires fewer stored training-image bytes than the conventional baseline to remain within a predeclared accuracy tolerance of its own uncompressed-training counterpart, while also meeting a shared absolute accuracy floor. A possible tolerance is one percentage point; this is a proposal to validate during the pilot, not a committed project requirement. The shared floor prevents a weak model from appearing successful merely because its clean baseline is low.

Report the entire accuracy-versus-bytes curve and the smallest tested storage budget that passes the criterion. An isolated success at one compression setting is weaker evidence. If no tested dictionary configuration improves the controlled trade-off, that is a meaningful negative result. Treat the 98% at greater-than-50:1 target as an unvalidated stretch target; the audit supplies no basis to promise it on an unspecified dataset.

For a one-month project, use one dataset, one conventional baseline, and one explicit dictionary module, with a small M-by-compression grid. Hold k, coding iterations, feature resolution, training schedule, augmentation, split identities, and tuning opportunities fixed. Add a parameter-matched dense control where feasible, since increasing M also increases parameters. Distinguish a clean-capacity effect from a change in sensitivity to compression.

Training inputs must actually come from the stored JPEG versions. Evaluate all models on the same held-out original test set; compressed-test robustness can be secondary. Select settings on validation data, then evaluate final comparisons with repeated seeds. If pretrained weights are used, describe savings for fine-tuning data, not all data involved in learning the model.

Record whole encoded image bytes, any shared codec tables/container overhead, dimensions, color mode, codec version/settings, and example count. Report both savings versus the actual original dataset files and a clearly labeled raw-RGB reference if useful. Do not silently discard storage overhead or change resolution to reach a target. A pixel-backed neural dictionary is not itself a new image compression format.

## Next evidence gaps

- Obtain Wei's primary full text before claiming a new dictionary-capacity interaction; check what is predicted, whether dictionary size is varied, and how training/test images are compressed.
- Compare Lee's final journal methods with arXiv v1. Do not attribute the proxy measurement to the final version until verified.
- Audit the existing Friedland 2020, Kuchnik 2021, and Pistono 2020 bibliography entries and follow relevant references/citations; this bounded pass is not exhaustive.
- Inspect an implementable sparse neural model and its license, optimizer, dictionary normalization, and sparsity mechanism. DDLCN abstract-level relevance does not guarantee a differentiable fixed-k module.
- Measure feasible complete-file rates at the chosen native image size before setting a 50:1 target. No such pilot was run here.

Novelty wording supported now: **“We investigate a controlled sparse-dictionary-capacity extension to existing compressed-training studies.”** Unsupported wording: “the first demonstration that extremely compressed images can be classified” or “no one has studied dictionary learning with compressed images.”
