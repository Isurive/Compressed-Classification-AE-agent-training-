"""Train a sparse dictionary with a decoder and deploy a classifier that has no decoder."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

from image_data import ImageView, reference_ids, training_splits
from resnet18_sparse_ae import SparseDictionaryAE, atomic_torch_save, load_trained, sparse_codes


PREDICTOR = "jpeg-sparse-resnet18/predictor-v1"
TRAINING = "jpeg-sparse-resnet18/training-v1"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


@dataclass
class Settings:
    data: str
    train_limit: int = 1000
    val_limit: int = 1000
    holdout: float = 0.2
    seed: int = 42
    atoms: int = 2048
    k: int = 32
    decoder_hidden: int = 0
    compression_level: int | None = 20
    batch_size: int = 64
    lr: float = 1e-3
    reconstruction_weight: float = 1.0
    ce_weight: float = 1.0
    dependency_weight: float = 0.1
    coherence_weight: float = 0.05
    label_space: str = "auto"
    cache: str = "auto"
    backbone_state: str | None = None
    init_sae: str | None = None
    random_backbone: bool = False
    encoder_init: str = "unit"
    fc_init: str = "auto"
    feature_mask: float = 0.0
    folds: int = 0
    fold: int = 1

    def validate(self):
        if min(self.train_limit, self.val_limit, self.decoder_hidden) < 0 or self.batch_size < 1:
            raise ValueError("Limits and decoder width must be nonnegative; batch size must be positive.")
        if not 0 < self.holdout < 1 or self.atoms < 2 or not 1 <= self.k <= self.atoms:
            raise ValueError("Use 0 < holdout < 1, atoms >= 2 and 1 <= k <= atoms.")
        values = [self.lr, self.ce_weight, self.reconstruction_weight, self.dependency_weight, self.coherence_weight]
        if not all(math.isfinite(v) for v in values) or min(values[:3]) <= 0 or min(values[3:]) < 0:
            raise ValueError("Learning rate, CE and reconstruction weights must be positive; penalties may be zero.")
        if self.encoder_init not in ("unit", "he") or self.fc_init not in ("auto", "folded", "xavier"):
            raise ValueError("Unknown encoder or classifier initialization.")
        if not 0 <= self.feature_mask < 1:
            raise ValueError("feature-mask must be in [0, 1).")
        if self.folds and (self.folds < 2 or not 1 <= self.fold <= self.folds):
            raise ValueError("Use at least two folds and a one-based fold number.")


class LatentClassifier(nn.Module):
    """The deployment graph. The only classifier is fc, which consumes the sparse code."""
    def __init__(self, backbone, projection, center, k, n_classes):
        super().__init__()
        backbone.fc = nn.Identity()
        self.backbone = backbone.requires_grad_(False).eval()
        self.encoder = projection
        self.center = center
        self.k = k
        self.fc = nn.Linear(projection.out_features, n_classes)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, images):
        return self.backbone(images)

    def encode_features(self, features):
        return sparse_codes(features, self.center, self.encoder, self.k)

    def encode(self, images):
        return self.encode_features(self.features(images))

    def forward(self, images):
        return self.fc(self.encode(images))


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") not in (PREDICTOR, TRAINING):
        raise ValueError("Expected best.pt or last.pt from main.py. Use train --init-sae for an earlier SAE.")
    return state


def restore_predictor(state, device):
    width, atoms = state["model"]["encoder.weight"].shape[1], state["atoms"]
    model = LatentClassifier(resnet18(weights=None), nn.Linear(width, atoms),
                             nn.Parameter(torch.zeros(width)), state["k"], len(state["classes"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def load_predictor(path, device="cpu"):
    """Importable inference API. Requires no downloads or decoder."""
    return restore_predictor(read_checkpoint(path), device)


def predictor_state(model, classes, ids, level, epoch, protocol, validation, standard_backbone):
    """Serialize exactly the modules used by direct inference."""
    return {"format": PREDICTOR, "model": model.state_dict(), "atoms": model.encoder.out_features,
            "k": model.k, "epoch": epoch, "classes": list(classes), "reference_ids": ids,
            "compression_level": level, "protocol": protocol, "validation": validation,
            "standard_backbone": standard_backbone}


def new_parts(cfg, n_classes):
    if cfg.init_sae:
        old = load_trained(cfg.init_sae)
        backbone, sae = old.backbone, old.sae
        if old.tuned_layer4 is not None:
            backbone.layer4 = old.tuned_layer4
        cfg.atoms, cfg.k = sae.encoder.out_features, sae.k
        cfg.decoder_hidden = sae.decoder_hidden
    else:
        weights = None if cfg.backbone_state or cfg.random_backbone else WEIGHTS
        if cfg.backbone_state:
            state = torch.load(cfg.backbone_state, map_location="cpu", weights_only=True)
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            if all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): val for key, val in state.items()}
            if "fc.weight" not in state or state["fc.weight"].shape[1] != 512:
                raise ValueError("The supplied checkpoint needs a TorchVision ResNet-18 with a 512-input fc.")
            backbone = resnet18(weights=None, num_classes=state["fc.weight"].shape[0])
            backbone.load_state_dict(state)
        else:
            backbone = resnet18(weights=weights)
        sae = SparseDictionaryAE(512, cfg.atoms, cfg.k, cfg.decoder_hidden, cfg.encoder_init)
    teacher_fc = backbone.fc.requires_grad_(False).eval()
    model = LatentClassifier(backbone, sae.encoder, sae.center, sae.k, n_classes)
    return model, sae, teacher_fc


def restore_training_parts(state, device):
    if state["format"] != TRAINING:
        raise ValueError("Resume requires last.pt, which contains the decoder and optimizer.")
    model = restore_predictor(state, device)
    cfg = state["settings"]
    sae = SparseDictionaryAE(512, state["atoms"], state["k"], cfg["decoder_hidden"]).to(device)
    sae.load_state_dict(state["sae"])
    # Share the trained encoder and center. Only the SAE owns decoder parameters.
    sae.encoder, sae.center = model.encoder, model.center
    teacher = nn.Linear(512, state["teacher"]["weight"].shape[0]).to(device)
    teacher.load_state_dict(state["teacher"])
    return model, sae, teacher.requires_grad_(False).eval()


@torch.no_grad()
def initialize_fc(model, sae, teacher, ids, mode="auto"):
    """Initialize the new latent-width classifier and return the strategy actually used."""
    if mode == "xavier":
        nn.init.xavier_uniform_(model.fc.weight, gain=1.0)
        nn.init.zeros_(model.fc.bias)
        return "xavier"
    if mode not in ("auto", "folded"):
        raise ValueError(f"Unknown classifier initialization: {mode}")
    if mode == "folded" and ids is None:
        raise ValueError("Folding requires a mapping to the original classifier rows. Use --fc-init xavier or auto.")
    if ids is not None:
        rows = torch.as_tensor(ids, device=teacher.weight.device)
        w = teacher.weight[rows]
        model.fc.weight.copy_(w @ sae.dictionary.T)
        model.fc.bias.copy_(teacher.bias[rows] + w @ sae.center)
        return "folded"
    return "pytorch-default"


def loader(dataset, batch, workers, device, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch, num_workers=workers, shuffle=shuffle,
                      pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed))


def batch_features(model, inputs, device):
    inputs = inputs.to(device, non_blocking=True)
    return inputs if inputs.ndim == 2 else model.features(inputs)


def use_memory_cache(mode, count):
    return mode == "memory" or (mode == "auto" and count <= 20000)


@torch.no_grad()
def maybe_cache(view, model, cfg, workers, device, name):
    if not use_memory_cache(cfg.cache, len(view)):
        return view
    print(f"Extracting frozen 512-D features once for {len(view):,} {name} images.", flush=True)
    features, labels = [], []
    for step, (images, target) in enumerate(loader(view, cfg.batch_size, workers, device), start=1):
        features.append(model.features(images.to(device, non_blocking=True)).cpu())
        labels.append(target)
        if step % 32 == 0:
            print(f"  {name}: {min(step * cfg.batch_size, len(view)):,}/{len(view):,}", flush=True)
    return TensorDataset(torch.cat(features), torch.cat(labels))


class FoldFeatureCache:
    """Share only fixed, per-image ResNet outputs across folds within one CV command."""
    def __init__(self):
        self.data = None

    def select(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        return TensorDataset(*(t.index_select(0, indices) for t in self.data.tensors))


class TrainingPass(NamedTuple):
    codes: torch.Tensor
    logits: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    ce: torch.Tensor
    loss: torch.Tensor


def training_pass(model, sae, features, labels, cfg):
    """Both supervised and reconstruction gradients flow through the same training code."""
    inputs = features
    if cfg.feature_mask:
        hidden = torch.rand_like(features) < cfg.feature_mask
        # Subtracting the learned center in encode_features turns hidden inputs into zero.
        inputs = torch.where(hidden, model.center, features)
    z = model.encode_features(inputs)
    logits = model.fc(z)
    reconstruction = sae.decode(z)
    mse = F.mse_loss(reconstruction, features)  # Full, unmasked reconstruction target.
    ce = F.cross_entropy(logits, labels)
    loss = cfg.reconstruction_weight * mse + cfg.ce_weight * ce
    if cfg.dependency_weight:
        loss = loss + cfg.dependency_weight * sae.dependency_loss(z)
    if cfg.coherence_weight:
        loss = loss + cfg.coherence_weight * sae.coherence_loss()
    return TrainingPass(z, logits, reconstruction, mse, ce, loss)


def training_epoch(model, sae, batches, optimizer, cfg, device):
    model.train()
    sae.train()
    totals = torch.zeros(4, dtype=torch.float32 if device.type == "mps" else torch.float64, device=device)
    count = 0
    for step, (inputs, labels) in enumerate(batches, start=1):
        features = batch_features(model, inputs, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        result = training_pass(model, sae, features, labels, cfg)
        result.loss.backward()
        optimizer.step()
        n = labels.numel()
        totals += torch.stack((result.loss.detach() * n, result.ce.detach() * n,
                               result.mse.detach() * n, (result.logits.argmax(1) == labels).sum()))
        count += n
        if step % 250 == 0:
            print(f"  train batch {step}/{len(batches)}   mean loss {totals[0].item() / count:.4f}", flush=True)
    averages = (totals / count).tolist()
    return dict(zip(("loss", "ce", "feature_mse", "top1"), averages))


@torch.inference_mode()
def evaluate(model, batches, device, sae=None, teacher=None, ids=None, *, compare_decoded=False):
    model.eval()
    n = 0
    sums = torch.zeros(8, dtype=torch.float32 if device.type == "mps" else torch.float64, device=device)
    firing = torch.zeros(model.encoder.out_features, dtype=torch.long, device=device)
    rows = torch.as_tensor(ids, device=device) if ids is not None and teacher is not None else None
    top = min(5, model.fc.out_features)
    for inputs, labels in batches:
        features = batch_features(model, inputs, device)
        labels = labels.to(device, non_blocking=True)
        z = model.encode_features(features)
        logits = model.fc(z)
        winners = logits.topk(top, dim=1).indices
        batch = labels.numel()
        sums[0] += (winners[:, 0] == labels).sum()
        sums[1] += (winners == labels[:, None]).any(dim=1).sum()
        sums[2] += F.cross_entropy(logits, labels, reduction="sum")
        firing += (z > 0).sum(dim=0)
        if sae is not None:
            reconstruction = sae.decode(z)
            sums[3] += F.mse_loss(reconstruction, features) * batch
            if compare_decoded and rows is not None:
                decoded_logits = teacher(reconstruction).index_select(1, rows)
                sums[7] += (decoded_logits.argmax(1) == labels).sum()
        if rows is not None:
            original = teacher(features).index_select(1, rows).topk(top, dim=1).indices
            sums[4] += (original[:, 0] == labels).sum()
            sums[5] += (original[:, 0] == winners[:, 0]).sum()
            sums[6] += (original == labels[:, None]).any(dim=1).sum()
        n += batch
    if not n:
        raise ValueError("Evaluation requires at least one image.")
    top1, topk, ce, mse, baseline, agreement, baseline_topk, decoded_top1 = (sums / n).tolist()
    result = {"images": n, "top1": top1, f"top{top}": topk, "ce": ce,
              "active_per_image": firing.sum().item() / n,
              "used_atom_fraction": (firing > 0).float().mean().item()}
    if sae is not None:
        result["feature_mse"] = mse
    if rows is not None:
        result.update(backbone_top1=baseline, top1_agreement=agreement)
        result[f"backbone_top{top}"] = baseline_topk
        if compare_decoded and sae is not None:
            result["decoded_backbone_top1"] = decoded_top1
    return result


@contextmanager
def run_writer(out):
    """Only one training process writes a run directory. The OS releases the lock on exit."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".training.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                if lock.tell() == 0:
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as exc:
            raise ValueError(f"Another process is writing {out}, or this directory cannot be locked.") from exc
        yield


def train(cfg, out, epochs, workers, device, resume=None, *, partitions=None, fold_cache=None):
    """Fit the encoder, decoder and direct classifier; return the complete metric history."""
    with run_writer(Path(out)):
        return _fit(cfg, Path(out), epochs, workers, device, resume, partitions, fold_cache)


def _fit(cfg, out, epochs, workers, device, resume, partitions, fold_cache):
    cfg.validate()
    if epochs < 1 or workers < 0:
        raise ValueError("Epochs must be positive and workers must be nonnegative.")
    best_path, last_path = out / "best.pt", out / "last.pt"
    if resume is None and (best_path.exists() or last_path.exists()):
        raise ValueError(f"{out} already has a run. Choose another --out or use resume.")
    first_epoch = resume["epoch"] + 1 if resume else 1
    if first_epoch > epochs:
        raise ValueError(f"Choose a total --epochs greater than the completed {first_epoch - 1}.")
    cfg.data = str(Path(cfg.data).expanduser().resolve())
    torch.manual_seed(cfg.seed)
    if partitions is None:
        partitions = training_splits(cfg.data, cfg.train_limit, cfg.val_limit, cfg.holdout, cfg.seed,
                                     cfg.folds, cfg.fold)
    tr, ti, va, vi, split, digest = partitions
    ids = reference_ids(tr.classes, cfg.label_space) if not cfg.random_backbone else None
    if resume:
        if resume["classes"] != tr.classes or resume["protocol"]["digest"] != digest:
            raise ValueError("The selected images, labels or order changed since this checkpoint.")
        model, sae, teacher = restore_training_parts(resume, device)
    else:
        model, sae, teacher = new_parts(cfg, len(tr.classes))
        model.to(device)
        sae.to(device)
        # Keep a single shared parameter even under parameter-replacing device conversions.
        sae.encoder, sae.center = model.encoder, model.center
        teacher.to(device)
    if teacher.out_features != 1000:
        ids = None
    print(f"{split}: {len(ti):,} training, {len(vi):,} validation, {len(tr.classes):,} classes. "
          f"Device {device}, backbone frozen, Adam lr={cfg.lr:g}.", flush=True)
    if ids is None:
        print("Using dataset labels for the new fc. ImageNet comparison is unavailable for this label space.", flush=True)
    if fold_cache is None:
        train_data = maybe_cache(ImageView(tr, ti, cfg.compression_level), model, cfg, workers, device, "training")
        val_data = maybe_cache(ImageView(va, vi, cfg.compression_level), model, cfg, workers, device, "validation")
    else:
        if fold_cache.data is None:
            all_images = ImageView(tr, range(len(tr.labels)), cfg.compression_level)
            fold_cache.data = maybe_cache(all_images, model, cfg, workers, device, "cross-validation pool")
        train_data, val_data = fold_cache.select(ti), fold_cache.select(vi)
    tw = 0 if isinstance(train_data, TensorDataset) else workers
    vw = 0 if isinstance(val_data, TensorDataset) else workers
    val_batches = loader(val_data, cfg.batch_size, vw, device)
    if not resume:
        if not cfg.init_sae:
            if isinstance(train_data, TensorDataset):
                center = train_data.tensors[0].mean(0).to(device)
            else:
                sample = next(iter(loader(train_data, cfg.batch_size, tw, device, shuffle=True, seed=cfg.seed)))[0]
                center = batch_features(model, sample, device).mean(0)
            with torch.no_grad():
                model.center.copy_(center)
        fc_strategy = initialize_fc(model, sae, teacher, ids, cfg.fc_init)
        print(f"Encoder initialization: {'checkpoint' if cfg.init_sae else cfg.encoder_init}. "
              f"Classifier initialization: {fc_strategy}. Training feature mask: {cfg.feature_mask:g}.", flush=True)
    optimizer = torch.optim.Adam([*sae.parameters(), *model.fc.parameters()], lr=cfg.lr)
    history = []
    best_score = (-1.0, float("-inf"))
    run_id = (resume.get("run_id") if resume else None) or uuid4().hex
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        history = list(resume["history"])
        if best_path.exists():
            incumbent = read_checkpoint(best_path)
            if (incumbent["format"] != PREDICTOR or incumbent["classes"] != tr.classes
                    or incumbent["protocol"].get("digest") != digest
                    or (resume.get("run_id") and incumbent.get("run_id", run_id) != run_id)):
                raise ValueError("The best.pt in this folder belongs to a different run or data split.")
            best_score = (incumbent["validation"]["top1"], -incumbent["validation"]["ce"])
            del incumbent
        else:
            # A copied last.pt is also a usable starting predictor for a new output directory.
            keys = ("model", "atoms", "k", "epoch", "classes", "reference_ids", "compression_level",
                    "protocol", "validation", "standard_backbone")
            snapshot = {k: resume[k] for k in keys if k in resume}
            snapshot["run_id"] = run_id
            atomic_torch_save(dict(snapshot, format=PREDICTOR), best_path)
            best_score = (resume["validation"]["top1"], -resume["validation"]["ce"])
    protocol = {"kind": split, "holdout": cfg.holdout, "seed": cfg.seed, "val_limit": cfg.val_limit, "digest": digest}
    if cfg.folds:
        protocol.update(folds=cfg.folds, fold=cfg.fold)
    for epoch in range(first_epoch, epochs + 1):
        # Seeding by epoch makes shuffle and sampled regularizers repeatable across resumes.
        torch.manual_seed(cfg.seed + epoch)
        batches = loader(train_data, cfg.batch_size, tw, device, shuffle=True, seed=cfg.seed + epoch)
        train_metrics = training_epoch(model, sae, batches, optimizer, cfg, device)
        metrics = evaluate(model, val_batches, device, sae, teacher, ids)
        if not all(math.isfinite(v) for v in (*train_metrics.values(), *metrics.values())):
            raise ValueError("Training produced non-finite metrics. Try a smaller learning rate.")
        record = {"epoch": epoch, "train": train_metrics, "validation": metrics}
        print(json.dumps(record, allow_nan=False), flush=True)
        history.append(record)
        common = predictor_state(model, tr.classes, ids, cfg.compression_level, epoch, protocol, metrics,
                                 not (cfg.init_sae or cfg.random_backbone or cfg.backbone_state))
        common["run_id"] = run_id
        score = (metrics["top1"], -metrics["ce"])
        if score > best_score:
            best_score = score
            atomic_torch_save(common, best_path)
        state = dict(common, format=TRAINING, settings=asdict(cfg), sae=sae.state_dict(),
                     teacher=teacher.state_dict(), optimizer=optimizer.state_dict(), history=history,
                     best_score=list(best_score))
        atomic_torch_save(state, last_path)
    print(f"Deployment: {best_path}   Resume and dictionary: {last_path}", flush=True)
    return history
