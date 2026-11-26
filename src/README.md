# JPEG, ResNet-18 and a sparse classifier

`main.py` attaches a sparse encoder and a feature decoder to a frozen ImageNet-1K ResNet-18. A new linear classifier reads the sparse code directly. Run `check` to verify the pipeline. Run `train` when you want to fit your own model.

## Start here

Use Python 3.10+ and a matching PyTorch/TorchVision installation. From the extracted directory, run the following in your terminal or activated Conda environment.

```bash
python -m pip install torch torchvision Pillow
python main.py check
```

Running `python main.py` alone also runs the check. The default check runs offline on four synthetic images. It performs JPEG preprocessing, one forward/backward pass and an in-memory checkpoint round trip. It checks gradients for the encoder, decoder and latent classifier, verifies that all weights and backbone BatchNorm buffers stay fixed, and compares predictions from the decoder-free export. The check takes zero optimizer steps. To check your local files and the ImageNet weights, run `python main.py check --data "/path/to/imagenet/val" --pretrained`.

## Train using your imagenet-1k/val directory

The commands below assume `main.py` and `imagenet-1k/` are in your current directory. If you extracted the ZIP beside an existing `imagenet-1k/` directory, stay in that parent directory and invoke `python jpeg_sparse_resnet/main.py` in place of `python main.py`. The standard validation pool has 50 images per class. A stratified 10% holdout takes five images per class for validation and leaves 45 for training. With 1,000 classes, that gives 45,000 training and 5,000 validation images.

```bash
# Preview the actual counts on disk. This reads labels and takes zero optimizer steps.
python main.py train --data ./imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 --batch-size 128 --epochs 10 --plan-only

# Fit the new encoder, decoder and classifier.
python main.py train --data ./imagenet-1k/val --holdout 0.10 --train-limit 0 --val-limit 0 --epochs 10 --batch-size 128 --lr 0.001 --cache memory --encoder-init he --fc-init xavier --out runs/he_45k

# Measure the saved best predictor and the original ResNet on the same held-out images.
python main.py evaluate --checkpoint runs/he_45k/best.pt --data ./imagenet-1k/val --limit 0 --baseline --json accuracy.json

# Continue the same split and optimizer to 20 total epochs.
python main.py resume --checkpoint runs/he_45k/last.pt --epochs 20
```

Both limits must be zero to remove the quick-run caps. At batch size 128, 45,000 images give 352 batches per epoch, including the last partial batch. Ten epochs give 3,520 Adam updates. The command uses the pretrained ImageNet-1K backbone, keeps it frozen and caches its pooled features in host memory across epochs. The 50,000 feature vectors themselves take about 98 MiB in float32, plus temporary buffers. Use `--cache off` to stream images. To compare against the previous initialization on exactly the same split, change only the flags of interest and choose another output directory. For the original pair, use `--encoder-init unit --fc-init auto`.

### Stratified 10-fold cross-validation

```bash
# Preview every fold without training.
python main.py crossval --data ./imagenet-1k/val --folds 10 --epochs 10 --batch-size 128 --plan-only

# Train all ten independent models.
python main.py crossval --data ./imagenet-1k/val --folds 10 --epochs 10 --batch-size 128 --lr 0.001 --cache memory --encoder-init he --fc-init xavier --out runs/cv10

# Evaluate only the images held out by fold 1.
python main.py evaluate --checkpoint runs/cv10/fold-01/best.pt --data ./imagenet-1k/val --limit 0 --baseline
```

Append `--fold 1` to the crossval command to fit only fold 1. Later run the all-fold command with `--resume` and the same options to continue partial folds and reuse completed folds. Each fold starts from fresh head parameters and Adam state. With the full pool, every image serves as validation once and training nine times, giving 35,200 updates across ten models at these settings. In a single all-fold command, `--cache memory` shares the fixed pretrained features. Each fold initializes its center using its own training subset.

The per-fold directories contain `best.pt` and `last.pt`. The root `cv_summary.json` reports pooled top-1 across held-out predictions, mean and standard deviation of fold top-1, and per-fold metrics at the requested epoch. Choosing the epoch in advance lets each model contribute at the same point in training. A single-fold invocation writes `fold_summary.json` in its fold directory. A model trained on part of the official validation set is measured as a local holdout or cross-validation experiment. Use a separate test set for an independent final score.

## Run the 1,000-image experiment

```bash
python main.py train --data "/path/to/imagenet" --out runs/quick
python main.py info --checkpoint runs/quick/last.pt --history
python main.py predict --checkpoint runs/quick/best.pt --image photo.jpg --save-latent photo_latent.pt
```

The default is 1,000 training images, up to 1,000 separate validation images, five epochs, batch size 64 and Adam with learning rate 0.001. Sampling is class-balanced with seed 42. With all 1,000 classes represented and sufficient examples per class, this gives one training example of each class. The backbone stays frozen, including BatchNorm statistics. The first standard training run downloads the TorchVision ImageNet-1K weights if they are missing from the local cache.

The trained predictor contains the frozen backbone and loads offline. Single-image prediction prints class probabilities and the strongest atoms. `--save-latent` saves the entire sparse vector under the key `latent`.

### Accepted dataset layouts

| Local data | Value for `--data` | How the split works |
| --- | --- | --- |
| `imagenet/train/<class>/...` and `imagenet/val/<class>/...` | `imagenet` | Use the separate folders |
| Only `imagenet/val/<class>/...` or a single directory with class subfolders | `imagenet/val` or its parent containing `val` | Hold out 20% within each class before sampling either subset |
| Original flat validation JPEGs named `ILSVRC2012_val_00000001.JPEG`, etc. | The directory containing those files | Look up each filename's official validation label, then split the labeled pool |
| A labeled Hugging Face `Dataset` or `DatasetDict` saved with `save_to_disk` | The saved directory | Use its train and validation splits when both exist, otherwise split the single labeled pool |
| Flat files with a CSV containing `path,label` | The CSV path | Resolve image paths relative to the CSV and split the labeled pool |

For the Hugging Face layout, install the optional reader with `python -m pip install datasets`. CSV manifests named `metadata.csv`, `labels.csv` or `manifest.csv` can also be discovered inside the supplied directory. Folder or CSV labels can be ImageNet synset IDs, zero-based ImageNet class IDs, or your own classes. For unrelated custom numeric labels, pass `--label-space custom`. For a full 1,000-class dataset with known canonical ordering and unrecognized names, pass `--label-space imagenet-order`.

When fitting on your downloaded validation pool, the scores describe that local holdout. For the standard ImageNet validation score, train using the separate training split. Tiny classes with only one example stay in the training pool during a single-folder split.

## What learns

1. Each source image goes through the JPEG encoder in memory, then through the matching TorchVision resize, crop and normalization. The default compression level is 20. `source` skips the extra JPEG pass. Level 0 still encodes a JPEG.
2. The frozen ResNet-18 produces a 512-value vector after global average pooling. The sparse encoder centers it, projects to 2,048 scores and keeps at most 32 positive values.
3. Two branches share that code during training. The decoder reconstructs the same compressed input's frozen 512-D feature vector. The trainable `Linear(2048, num_classes)` predicts the label directly. PyTorch autograd updates the encoder, center, decoder and new classifier.
4. The default loss is `cross_entropy + feature_MSE + 0.1 * decoder_dependency + 0.05 * atom_coherence`. Dependency sparsity encourages each atom to affect fewer output channels. The coherence term discourages nearly parallel or opposite dictionary atoms. To use a nonlinear decoder residual, set `--decoder-hidden 128`.
5. At prediction time the graph ends with `encoder → sparse code → new fc`. The deployment checkpoint contains the backbone, sparse encoder, center and new fc. The `encode` command stops at the latent vector.

By default, initialization reuses the original classifier for ImageNet-mapped classes. With dictionary rows `D`, center `c`, old classifier weights `W`, and bias `b`, the new fc starts at `W @ D.T` and `b + W @ c`. For a linear decoder, this gives the same initial logits as explicitly decoding and applying the old classifier. Cross-entropy then updates the new fc independently. Custom classes start with a freshly initialized fc.

### Encoder, decoder and classifier initialization

| Component | Shape for ImageNet-1K | Choices |
| --- | --- | --- |
| SAE encoder | 512 inputs → 2,048 scores | The default `--encoder-init unit` draws unit-length random rows. `he` draws independent He-normal weights with standard deviation `sqrt(2 / 512) = 0.0625`. Both start with zero bias. |
| Linear decoder | 2,048 codes → 512 features | Each dictionary atom starts in its corresponding encoder row's direction and has unit length. The decoder and encoder can then learn independently. |
| Classification head | 2,048 codes → 1,000 logits | `--fc-init auto` folds the old weights into this wider layer for known ImageNet classes and uses PyTorch's linear default otherwise. `folded` requires an ImageNet class mapping. `xavier` initializes the entire new matrix independently with Xavier uniform weights and zero bias. |

These options apply when starting a run. Resuming restores the trained parameters. Importing an existing SAE with `--init-sae` keeps its encoder and dictionary. The classification head in the integrated workflow is always allocated for the latent width.

### What prevents collapse and coordinate copying?

All-ones encoder rows give the same sum in every score, and duplicate decoder atoms can reconstruct only one direction around the center. An all-zero code yields a single reconstruction and one set of class logits for every image. Feature MSE and label cross-entropy reward retaining differences between images. The top-K gate limits every image to at most 32 positive coefficients, and the coherence penalty increases when different atoms point in nearly the same or opposite directions.

An identity matrix has diagonal ones and off-diagonal zeros. Some learned atoms can legitimately resemble individual ResNet channels. The decoder dependency term favors atoms concentrated on fewer channels. A loss pushing every weight away from zero would still allow duplicate dense atoms. With a hard top-K gate, an encoder row that never wins gets no gradient from the reconstruction or classification loss through that row. The reported usage fraction counts atoms selected at least once, so also compare held-out MSE and top-1, and inspect activation frequencies in exported codes when investigating rarely used atoms. These objectives do not guarantee human-interpretable features.

To test how much the model relies on copying visible coordinates, add `--feature-mask 0.15` to a fresh run. On each training example random channels are replaced by the center before encoding, and the decoder reconstructs the full original feature vector. Inference uses all channels. The default is 0. A controlled comparison uses the same split, epochs, seed and initialization across two output directories. To isolate initialization effects, vary one of `--encoder-init` and `--fc-init` at a time. You can check the new paths without training with `python main.py check --encoder-init he --fc-init xavier --feature-mask 0.15`.

## Files in this project

| File | Role |
| --- | --- |
| `main.py` | CLI for data planning, training, cross-validation, resuming, evaluating, predicting, exporting codes, JPEG conversion, checkpoint inspection and verification. |
| `sparse_workflow.py` | Connects the frozen backbone, SAE and direct latent classifier. Owns the loss, autograd loop, evaluation, feature cache, checkpoints and decoder-free inference API. |
| `resnet18_sparse_ae.py` | Implements the top-K encoder, dictionary decoder and regularizers. Also includes the original standalone wrapper, which applies a classifier to reconstructed features. Use `main.py` for direct latent classification. |
| `image_data.py` | Reads labeled folders, manifests and saved datasets, aligns class IDs, creates reproducible splits, and applies JPEG and ResNet preprocessing. |
| `compress_jpeg.py` | Encodes a file or an in-memory image. Higher compression levels use lower JPEG quality. |
| `imagenet1k_wnids.txt` | Lists ImageNet synset IDs in the original classifier's row order. |
| `imagenet1k_val_labels.txt.gz` | Supplies labels for the standard flat validation JPEG filenames. |
| `README.md` | Contains setup, training, cross-validation, inference and evaluation commands. |

## Train larger models and resume

```bash
python main.py train --data "/path/to/imagenet" --out runs/full --train-limit 0 --val-limit 0 --epochs 15 --atoms 4096 --k 64 --batch-size 128 --device cuda --workers 4
python main.py resume --checkpoint runs/quick/last.pt --epochs 10
```

A limit of 0 selects all eligible images. Resume uses the saved model, decoder, optimizer, compression level and split settings. `--epochs 10` means ten total. If the same dataset moved, pass `--data NEW_PATH`. Checkpoints are written after each epoch. On small splits the runner caches frozen features in memory and reuses them across epochs. With `--cache auto`, splits above 20,000 images stream from disk. Override using `--cache off` or `--cache memory`.

| Output | Contents and use |
| --- | --- |
| `best.pt` | Decoder-free predictor selected by highest validation top-1, with lower validation cross-entropy as the tie-breaker |
| `last.pt` | Latest predictor, decoder and dictionary, original fc, Adam state, epoch history and data split fingerprint |

## Evaluation and compression sweeps

```bash
python main.py evaluate --checkpoint runs/quick/last.pt --data "/path/to/imagenet" --levels source 0 20 70 95 --reconstruction --json metrics.json
python main.py evaluate --checkpoint runs/quick/best.pt --data "/path/to/independent_test" --entire-input --limit 0 --baseline
```

Every compression level uses the same images. Accuracy always comes from the direct latent fc. Metrics include top-1, top-5 (or the available class count), cross-entropy, active atoms per image and the fraction of dictionary atoms used. `--reconstruction` adds feature MSE and needs `last.pt`. For mapped ImageNet labels, `last.pt` also gives original-backbone top-1 and agreement on the same JPEG inputs. `--baseline` on a standard `best.pt` loads the original TorchVision classifier. For an ImageNet class subset, both classifiers compete over those same classes. Accuracy and agreement are fractions from 0 to 1.

Use `--entire-input` only when supplying a genuinely separate test set. With a single training/validation pool, the default evaluation reconstructs its held-out split. `--limit 0` selects every eligible image for the requested split.

### Evaluate an existing 45k / 5k checkpoint

```bash
python main.py info --checkpoint runs/holdout45k/last.pt --history
python main.py evaluate --checkpoint runs/holdout45k/last.pt --data ./imagenet-1k/val --limit 0 --baseline --reconstruction --json runs/holdout45k/heldout_epoch10.json
```

The first command shows the epoch, split settings and per-epoch metrics. The second uses every held-out image for the epoch saved in `last.pt`. Training for ten epochs reuses the same training subset each epoch. A 1,000-image evaluation samples the saved holdout. The full 5,000-image evaluation also includes those 1,000. Validation images help select `best.pt` and any settings chosen after looking at validation results.

Compare three top-1 values on the same inputs at the saved JPEG level. `backbone_top1` runs the original fc on the original frozen features. `decoded_backbone_top1` runs that same fc on the reconstructed features. A drop between those two measures how reconstruction changes the original classifier's accuracy. `top1` runs your learned fc directly on the latent code. If reconstruction preserves most of the original accuracy and the learned fc scores lower, investigate the learned classifier. Normal prediction and code export use the encoder directly.

For an independent final test, choose the checkpoint using validation, then use a separately labeled set with the same classes. ImageNetV2 MatchedFrequency provides 10,000 separately collected ImageNet-class images. Extract the folder-format archive and point `--data` to the directory containing the numeric class subdirectories. The example below assumes you named that directory `imagenetv2-matched-frequency`. [ImageNetV2 data and loading instructions](https://github.com/modestyachts/ImageNetV2#loading-a-test-set)

```bash
python main.py evaluate --checkpoint runs/holdout45k/best.pt --data ./imagenetv2-matched-frequency --entire-input --limit 0 --baseline --json runs/holdout45k/imagenetv2.json
```

## Export codes or use an earlier model

```bash
python main.py encode --checkpoint runs/quick/best.pt --input ./photos --output latents.csv
python main.py compress input.png output.jpg --compression-level 70
python main.py predict --checkpoint runs/quick/best.pt --image output.jpg --compression-level source
python main.py train --data "/path/to/imagenet" --backbone-state resnet18_weights.pt --out runs/my_backbone
python main.py train --data "/path/to/imagenet" --init-sae resnet18_sae.pt --out runs/warm_start
python main.py convert --sae resnet18_sae.pt --output folded.pt
```

The CSV exporter processes a file or recursive directory in batches. Each row records the relative filename, latent width, atom indices and matching positive values. All other entries of that vector are zero. The old linear SAE can be folded exactly into a direct classifier with `convert`. For an older nonlinear SAE, use `train --init-sae` to learn the direct fc. The warm start freezes its effective feature extractor, including a previously tuned last stage.

### Python inference

```python
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights
from compress_jpeg import model_image
from sparse_workflow import read_checkpoint, restore_predictor

state = read_checkpoint("runs/quick/best.pt")
model = restore_predictor(state, "cpu")
with Image.open("photo.jpg") as image:
    rgb = model_image(image, state["compression_level"])
    x = ResNet18_Weights.IMAGENET1K_V1.transforms()(rgb).unsqueeze(0)
with torch.inference_mode():
    z = model.encode(x)       # [1, n_atoms]
    logits = model.fc(z)      # [1, num_classes]
```

The original standalone JPEG and SAE scripts are also included. For every CLI option, use `python main.py COMMAND --help`.

The label lookup tables come from TensorFlow Datasets. The decoder regularizer adapts dependency sparsity from Zheng et al. [ImageNet synset ordering](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/imagenet2012/labels.txt) · [Validation labels](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/imagenet2012/validation_labels.txt) · [Diverse Dictionary Learning](https://arxiv.org/abs/2604.17568)
