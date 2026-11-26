"""Conventional baseline: a linear classifier on the same frozen ResNet-18 features.

This is the "no dictionary" comparison for main.py. It reuses main.py's split, JPEG
preprocessing, feature cache, data loader, seeds and best-checkpoint rule unchanged,
so the only difference is the head:

    main.py (sparse):   512-d feature -> encoder -> top-k code (M atoms, k active) -> fc
                        trained with CE + feature MSE + dependency + coherence
    main.py --k M:      same, but every atom may be active (dense control: no sparsity)
    baseline_head.py:   512-d feature -> Linear(512, classes), trained with CE only

Train and evaluate with the same options you used for main.py:

    python baseline_head.py train --data ../Datasets/imagenet-1k/val --holdout 0.10 \
        --train-limit 0 --val-limit 0 --epochs 10 --batch-size 128 --lr 0.001 \
        --compression-level 20 --cache memory --workers 4 --out ../runs/linear_45k
    python baseline_head.py evaluate --checkpoint ../runs/linear_45k/best.pt \
        --data ../Datasets/imagenet-1k/val --limit 0 --levels source 20 --json metrics.json

The split digest printed at the start must equal the one in the main.py checkpoint you
compare against (python main.py info --checkpoint ...). If it differs, the images differ.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.models import resnet18

from image_data import ImageView, evaluation_split, reference_ids, training_splits
from main import device_for, jpeg_level, text_output
from resnet18_sparse_ae import atomic_torch_save
from sparse_workflow import WEIGHTS, batch_features, loader, maybe_cache, run_writer

FORMAT = "jpeg-sparse-resnet18/linear-baseline-v1"


class FrozenFeatures(nn.Module):
    """The same frozen ImageNet ResNet-18 that main.py uses, ending at the 512-d pooled feature."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=WEIGHTS)
        self.teacher = backbone.fc.requires_grad_(False)
        backbone.fc = nn.Identity()
        self.backbone = backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, images):
        return self.backbone(images)


@torch.inference_mode()
def evaluate(model, head, batches, device, ids):
    """Top-1, top-5 and CE of the head. Adds the original ImageNet fc on the same inputs when mapped."""
    head.eval()
    rows = torch.as_tensor(ids, device=device) if ids is not None else None
    top = min(5, head.out_features)
    n, sums = 0, torch.zeros(5, dtype=torch.float64, device=device)
    for inputs, labels in batches:
        features = batch_features(model, inputs, device)
        labels = labels.to(device, non_blocking=True)
        logits = head(features)
        winners = logits.topk(top, dim=1).indices
        sums[0] += (winners[:, 0] == labels).sum()
        sums[1] += (winners == labels[:, None]).any(dim=1).sum()
        sums[2] += F.cross_entropy(logits, labels, reduction="sum")
        if rows is not None:
            original = model.teacher(features).index_select(1, rows).topk(top, dim=1).indices
            sums[3] += (original[:, 0] == labels).sum()
            sums[4] += (original == labels[:, None]).any(dim=1).sum()
        n += labels.numel()
    if not n:
        raise ValueError("Evaluation requires at least one image.")
    top1, topk, ce, base1, basek = (sums / n).tolist()
    result = {"images": n, "top1": top1, f"top{top}": topk, "ce": ce}
    if rows is not None:
        result.update(backbone_top1=base1, **{f"backbone_top{top}": basek})
    return result


def train(args, device):
    out = Path(args.out)
    with run_writer(out):
        best_path = out / "best.pt"
        if best_path.exists() or (out / "history.json").exists():
            raise ValueError(f"{out} already has a run. Choose another --out.")
        data = str(Path(args.data).expanduser().resolve())
        torch.manual_seed(args.seed)
        tr, ti, va, vi, split, digest = training_splits(data, args.train_limit, args.val_limit,
                                                        args.holdout, args.seed)
        ids = reference_ids(tr.classes, args.label_space)
        model = FrozenFeatures().to(device)
        head = nn.Linear(512, len(tr.classes)).to(device)
        if args.fc_init == "xavier":
            nn.init.xavier_uniform_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
        elif args.fc_init == "original":
            if ids is None:
                raise ValueError("--fc-init original needs a mapping to ImageNet classes.")
            with torch.no_grad():
                rows = torch.as_tensor(ids, device=device)
                head.weight.copy_(model.teacher.weight[rows])
                head.bias.copy_(model.teacher.bias[rows])
        print(f"{split}: {len(ti):,} training, {len(vi):,} validation, {len(tr.classes):,} classes. "
              f"Device {device}, backbone frozen, head Linear(512, {len(tr.classes)}) init {args.fc_init}, "
              f"Adam lr={args.lr:g} weight_decay={args.weight_decay:g}. Split digest {digest}", flush=True)
        cache_cfg = SimpleNamespace(cache=args.cache, batch_size=args.batch_size)
        train_data = maybe_cache(ImageView(tr, ti, args.compression_level), model, cache_cfg, args.workers, device, "training")
        val_data = maybe_cache(ImageView(va, vi, args.compression_level), model, cache_cfg, args.workers, device, "validation")
        tw = 0 if isinstance(train_data, TensorDataset) else args.workers
        vw = 0 if isinstance(val_data, TensorDataset) else args.workers
        val_batches = loader(val_data, args.batch_size, vw, device)
        optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        protocol = {"kind": split, "holdout": args.holdout, "seed": args.seed, "val_limit": args.val_limit,
                    "digest": digest}
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "command"}
        history, best_score = [], (-1.0, float("-inf"))
        for epoch in range(1, args.epochs + 1):
            torch.manual_seed(args.seed + epoch)  # Same per-epoch shuffle seeding as main.py.
            head.train()
            n, loss_sum, correct = 0, 0.0, 0
            for inputs, labels in loader(train_data, args.batch_size, tw, device, shuffle=True, seed=args.seed + epoch):
                features = batch_features(model, inputs, device)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = head(features)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.numel()
                correct += (logits.argmax(1) == labels).sum().item()
                n += labels.numel()
            metrics = evaluate(model, head, val_batches, device, ids)
            record = {"epoch": epoch, "train": {"ce": loss_sum / n, "top1": correct / n}, "validation": metrics}
            if not all(math.isfinite(v) for v in (*record["train"].values(), *metrics.values())):
                raise ValueError("Training produced non-finite metrics. Try a smaller learning rate.")
            print(json.dumps(record), flush=True)
            history.append(record)
            score = (metrics["top1"], -metrics["ce"])  # Same selection rule as main.py's best.pt.
            if score > best_score:
                best_score = score
                atomic_torch_save({"format": FORMAT, "head": head.state_dict(), "classes": list(tr.classes),
                                   "reference_ids": ids, "compression_level": args.compression_level,
                                   "epoch": epoch, "protocol": protocol, "validation": metrics,
                                   "settings": settings}, best_path)
        with text_output(out / "history.json") as f:
            json.dump({"settings": settings, "protocol": protocol, "history": history}, f, indent=2)
        print(f"Best checkpoint: {best_path}", flush=True)


def run_evaluation(args, device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("Expected best.pt from baseline_head.py. Use main.py evaluate for sparse checkpoints.")
    protocol = state["protocol"]
    limit = protocol["val_limit"] if args.limit is None else args.limit
    # Same held-out selection as main.py evaluate.
    source, idx = evaluation_split(args.data, state["classes"], protocol, protocol["kind"], limit,
                                   args.entire_input, state["reference_ids"])
    model = FrozenFeatures().to(device)
    head = nn.Linear(512, len(state["classes"])).to(device)
    head.load_state_dict(state["head"])
    rows = []
    for level in (args.levels if args.levels is not None else [state["compression_level"]]):
        metrics = evaluate(model, head, loader(ImageView(source, idx, level), args.batch_size, args.workers, device),
                           device, state["reference_ids"])
        row = {"compression_level": "source" if level is None else level, **metrics}
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.json:
        with text_output(args.json) as f:
            json.dump(rows, f, indent=2)


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="Fit a linear head on frozen features, using main.py's split and budget.")
    t.add_argument("--data", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--train-limit", type=int, default=1000, help="Balanced training count. 0 uses all.")
    t.add_argument("--val-limit", type=int, default=1000, help="Balanced validation count. 0 uses all.")
    t.add_argument("--holdout", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--compression-level", type=jpeg_level, default=20, help="0..100, or source.")
    t.add_argument("--fc-init", choices=("xavier", "original"), default="xavier",
                   help="xavier matches main.py --fc-init xavier; original copies the ImageNet fc rows.")
    t.add_argument("--cache", choices=("auto", "memory", "off"), default="auto")
    t.add_argument("--label-space", choices=("auto", "custom", "imagenet-order"), default="auto")
    e = commands.add_parser("evaluate", help="Score a baseline checkpoint on main.py's held-out images.")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--limit", type=int, help="Defaults to the saved validation count. 0 uses all.")
    e.add_argument("--levels", type=jpeg_level, nargs="+")
    e.add_argument("--entire-input", action="store_true", help="Treat a separate folder as the test set.")
    e.add_argument("--json", type=Path)
    for sub in (t, e):
        sub.add_argument("--device", default="auto")
        sub.add_argument("--workers", type=int, default=0)
        sub.add_argument("--threads", type=int, default=0)
    return parser


def main():
    parser = cli()
    args = parser.parse_args()
    try:
        if min(getattr(args, "epochs", 1), args.batch_size) < 1:
            raise ValueError("epochs and batch-size must be positive.")
        device = device_for(args)
        (train if args.command == "train" else run_evaluation)(args, device)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
