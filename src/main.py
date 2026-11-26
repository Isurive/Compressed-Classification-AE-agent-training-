"""JPEG → frozen ResNet-18 → sparse code → classifier. Run python main.py --help."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from compress_jpeg import compress_jpeg, model_image
from image_data import ImageView, UnlabeledImages, evaluation_split, reference_ids, sources, training_splits
from resnet18_sparse_ae import atomic_torch_save, load_trained
from sparse_workflow import (
    TRAINING, WEIGHTS, FoldFeatureCache, LatentClassifier, Settings, evaluate, initialize_fc, load_predictor,
    loader, new_parts, predictor_state, read_checkpoint, restore_predictor, restore_training_parts, train,
    run_writer, training_pass, use_memory_cache,
)


def jpeg_level(text):
    if str(text).lower() == "source":
        return None
    try:
        n = int(text)
    except ValueError:
        n = -1
    if not 0 <= n <= 100:
        raise argparse.ArgumentTypeError("Use source or an integer from 0 through 100.")
    return n


def device_for(args):
    name = args.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation. Use --device cpu or install a CUDA build.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch installation.")
    if args.workers < 0 or args.threads < 0:
        raise ValueError("workers and threads must be nonnegative.")
    if args.threads or device.type == "cpu":
        torch.set_num_threads(args.threads or min(8, torch.get_num_threads()))
    return device


@contextmanager
def text_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", newline="", delete=False) as f:
            temp_path = Path(f.name)
            yield f
        os.replace(temp_path, path)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def input_level(args, checkpoint):
    return getattr(args, "compression_level", checkpoint["compression_level"])


def checked_output(path, inputs):
    path = Path(path).resolve()
    if any(path == Path(item).resolve() for item in inputs if item is not None):
        raise ValueError("Choose an output path different from the input and checkpoint.")
    return path


def class_name(state, index):
    ids = state["reference_ids"]
    return WEIGHTS.meta["categories"][ids[index]] if ids is not None else state["classes"][index]


@torch.inference_mode()
def predict(args, device):
    state = read_checkpoint(args.checkpoint)
    model = restore_predictor(state, device)
    if not args.image.is_file() or args.top < 1:
        raise ValueError("predict requires a single image file and --top >= 1.")
    image, _ = UnlabeledImages(args.image, input_level(args, state))[0]
    z = model.encode(image.unsqueeze(0).to(device))
    probabilities = model.fc(z).softmax(1)[0]
    scores, indices = probabilities.topk(min(args.top, len(probabilities)))
    result = [{"class": class_name(state, i), "class_index": i, "probability": p}
              for p, i in zip(scores.tolist(), indices.tolist())]
    values, atoms = z[0].topk(model.k)
    active = [(j, v) for j, v in zip(atoms.tolist(), values.tolist()) if v > 0]
    print(json.dumps({"predictions": result, "latent_width": z.shape[1], "active_atoms": len(active),
                      "strongest_atoms": active[:8]}, indent=2))
    if args.save_latent:
        dest = checked_output(args.save_latent, [args.checkpoint, args.image])
        atomic_torch_save({"latent": z[0].cpu(), "k": model.k}, dest)
        print(f"Saved latent tensor to {dest}")


@torch.inference_mode()
def encode(args, device):
    state = read_checkpoint(args.checkpoint)
    model = restore_predictor(state, device)
    data = UnlabeledImages(args.input, input_level(args, state))
    dest = checked_output(args.output, [args.checkpoint, args.input])
    if dest.suffix.lower() != ".csv":
        raise ValueError("Use a .csv output file for the exported sparse codes.")
    with text_output(dest) as f:
        writer = csv.writer(f)
        writer.writerow(("image", "latent_width", "atom_indices", "activation_values"))
        for images, paths in loader(data, args.batch_size, args.workers, device):
            codes = model.encode(images.to(device))
            values, atoms = codes.topk(model.k, dim=1)
            for path, idx, vals in zip(paths, atoms.tolist(), values.tolist()):
                positive = [(i, v) for i, v in zip(idx, vals) if v > 0]
                writer.writerow((path, codes.shape[1], json.dumps([i for i, _ in positive]),
                                 json.dumps([v for _, v in positive])))
    print(f"Encoded {len(data):,} images to {dest}")


def run_evaluation(args, device):
    state = read_checkpoint(args.checkpoint)
    teacher = sae = None
    if args.reconstruction:
        if state["format"] != TRAINING:
            raise ValueError("Feature reconstruction requires last.pt. best.pt contains only the deployment model.")
        model, sae, teacher = restore_training_parts(state, device)
    else:
        model = restore_predictor(state, device)
    if teacher is None and state["format"] == TRAINING:
        teacher = nn.Linear(512, state["teacher"]["weight"].shape[0]).to(device)
        teacher.load_state_dict(state["teacher"])
    if args.baseline and teacher is None:
        if not state.get("standard_backbone"):
            raise ValueError("Use this run's last.pt for its original classifier. --baseline on best.pt requires standard weights.")
        teacher = resnet18(weights=WEIGHTS).fc.to(device)
    if args.baseline and state["reference_ids"] is None:
        raise ValueError("The checkpoint has no mapping from this dataset's classes to ImageNet labels.")
    protocol = state["protocol"]
    limit = protocol["val_limit"] if args.limit is None else args.limit
    if limit < 0:
        raise ValueError("--limit must be >= 0. Use 0 for all eligible images.")
    source, idx = evaluation_split(args.data, state["classes"], protocol, protocol["kind"],
                                   limit, args.entire_input, state["reference_ids"])
    rows = []
    levels = args.levels if args.levels is not None else [state["compression_level"]]
    for level in levels:
        data = ImageView(source, idx, level)
        metrics = evaluate(model, loader(data, args.batch_size, args.workers, device), device,
                           sae, teacher, state["reference_ids"], compare_decoded=args.reconstruction)
        row = {"compression_level": "source" if level is None else level, **metrics}
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.json:
        with text_output(checked_output(args.json, [args.checkpoint, args.data])) as f:
            json.dump(rows, f, indent=2)


def export_legacy(args):
    """Fold a previously trained linear decoder and its fc exactly for direct inference."""
    old = load_trained(args.sae)
    if old.sae.residual_decoder is not None:
        raise ValueError("Exact folding requires a linear decoder. Run train --init-sae for this nonlinear checkpoint.")
    if old.tuned_layer4 is not None:
        old.backbone.layer4 = old.tuned_layer4
    teacher = old.backbone.fc
    n = teacher.out_features
    model = LatentClassifier(old.backbone, old.sae.encoder, old.sae.center, old.sae.k, n)
    initialize_fc(model, old.sae, teacher, list(range(n)))
    protocol = {"kind": "external", "seed": 42, "holdout": 0.2, "val_limit": 1000}
    state = predictor_state(model, list(map(str, range(n))), list(range(n)) if n == 1000 else None,
                            args.compression_level, 0, protocol, {}, False)
    atomic_torch_save(state, checked_output(args.output, [args.sae]))
    print(f"Saved encoder and folded classifier to {args.output}")


def check_pipeline(args, device):
    """Exercise the actual model and deployment loader without taking an optimizer step."""
    torch.manual_seed(42)
    cfg = Settings(data=str(args.data or ""), atoms=args.atoms, k=args.k, decoder_hidden=args.decoder_hidden,
                   compression_level=args.compression_level, batch_size=args.batch_size, random_backbone=not args.pretrained,
                   encoder_init=args.encoder_init, fc_init=args.fc_init, feature_mask=args.feature_mask)
    cfg.validate()
    if args.data:
        source, ids, _, _, kind, _ = training_splits(args.data, args.batch_size, args.batch_size, 0.2, 42)
        classes = source.classes
        images, labels = next(iter(loader(ImageView(source, ids, cfg.compression_level),
                                          args.batch_size, args.workers, device)))
    else:
        kind, classes = "synthetic", list(map(str, range(1000)))
        pixels = torch.randint(256, (args.batch_size, 96, 128, 3), dtype=torch.uint8)
        transform = WEIGHTS.transforms()
        images = torch.stack([transform(model_image(Image.fromarray(row.numpy()), cfg.compression_level))
                              for row in pixels])
        labels = torch.arange(args.batch_size) % len(classes)
    model, sae, teacher = new_parts(cfg, len(classes))
    model.to(device)
    sae.to(device)
    sae.encoder, sae.center = model.encoder, model.center
    mapping = reference_ids(classes, "auto") if args.pretrained or cfg.fc_init == "folded" else None
    fc_strategy = initialize_fc(model, sae, teacher.to(device), mapping, cfg.fc_init)
    model.train()
    tensors = list(dict.fromkeys([*model.parameters(), *sae.parameters(), *model.backbone.buffers()]))
    before = [value.detach().cpu().clone() for value in tensors]
    images, labels = images.to(device), labels.to(device)
    features = model.features(images)
    result = training_pass(model, sae, features, labels, cfg)
    result.loss.backward()
    for name, param in (("encoder", model.encoder.weight), ("decoder", sae.atom_weights), ("latent fc", model.fc.weight)):
        if param.grad is None or not torch.isfinite(param.grad).all() or param.grad.abs().sum() == 0:
            raise RuntimeError(f"Missing, zero or non-finite {name} gradient.")
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("A frozen backbone parameter received a gradient.")
    if any(not torch.equal(v.detach().cpu(), original) for v, original in zip(tensors, before)):
        raise RuntimeError("A weight or BatchNorm buffer changed during the check.")
    if (result.codes < 0).any() or (result.codes.count_nonzero(dim=1) > cfg.k).any():
        raise RuntimeError("The encoder violated its sparse-code constraint.")
    del before
    protocol = {"kind": kind, "seed": 42, "holdout": 0.2, "val_limit": args.batch_size}
    state = predictor_state(model, classes, mapping, cfg.compression_level, 0, protocol, {}, args.pretrained)
    if any("atom_weights" in key or "residual_decoder" in key for key in state["model"]):
        raise RuntimeError("The deployment state includes decoder weights.")
    with BytesIO() as stream:
        torch.save(state, stream)
        stream.seek(0)
        deployed = load_predictor(stream, device)
    with torch.inference_mode():
        z = model.encode_features(features)  # Evaluation always uses all input channels.
        logits = model.fc(z)
        torch.testing.assert_close(deployed.encode(images), z)
        torch.testing.assert_close(deployed(images), logits)
    print(json.dumps({"result": "PASS", "input": kind, "batch": len(labels),
                      "backbone_features": list(features.shape), "sparse_codes": list(z.shape),
                      "encoder_init": cfg.encoder_init, "fc_init": fc_strategy, "train_feature_mask": cfg.feature_mask,
                      "max_active": int(z.count_nonzero(dim=1).max()), "reconstruction": list(result.reconstruction.shape),
                      "classifier_logits": list(logits.shape), "finite_gradients": ["encoder", "decoder", "latent_fc"],
                      "frozen_backbone": True, "parameters_unchanged": True, "optimizer_steps": 0,
                      "decoder_free_reload_matches": True}, indent=2))


def runtime(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default 0).")
    parser.add_argument("--threads", type=int, default=0, help="CPU compute threads. 0 chooses up to 8 for CPU runs.")


def initialization_options(parser):
    parser.add_argument("--encoder-init", choices=("unit", "he"), default="unit",
                        help="Unit-length or He-normal encoder rows, aligned with the initial decoder atoms.")
    parser.add_argument("--fc-init", choices=("auto", "folded", "xavier"), default="auto",
                        help="auto folds a mapped pretrained fc. xavier initializes the new classifier independently.")
    parser.add_argument("--feature-mask", type=float, default=0.0,
                        help="Probability of hiding each input feature during fitting. The target uses all features.")


def fitting_options(t, cv=False):
    runtime(t)
    initialization_options(t)
    t.add_argument("--data", required=True, help="Local class folders, saved HF dataset or path/label CSV.")
    t.add_argument("--out", type=Path, default=Path("runs/cv" if cv else "runs/quick"))
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--train-limit", type=int, default=0 if cv else 1000, help="Balanced training count. 0 uses all.")
    t.add_argument("--val-limit", type=int, default=0 if cv else 1000, help="Balanced validation count. 0 uses all.")
    t.add_argument("--plan-only", action="store_true", help="Print exact split sizes and update counts without constructing a model.")
    if not cv:
        t.add_argument("--holdout", type=float, default=0.2, help="Per-class holdout fraction for a single labeled pool.")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--atoms", type=int, default=2048)
    t.add_argument("--k", type=int, default=32)
    t.add_argument("--decoder-hidden", type=int, default=0, help="0 for a linear decoder; positive for a nonlinear residual.")
    t.add_argument("--compression-level", type=jpeg_level, default=20, help="0..100, or source to skip re-encoding.")
    t.add_argument("--ce-weight", type=float, default=1.)
    t.add_argument("--reconstruction-weight", type=float, default=1.)
    t.add_argument("--dependency-weight", type=float, default=.1)
    t.add_argument("--coherence-weight", type=float, default=.05)
    t.add_argument("--cache", choices=("auto", "memory", "off"), default="auto", help="Cache frozen features for repeated epochs.")
    t.add_argument("--label-space", choices=("auto", "custom", "imagenet-order"), default="auto")
    init = t.add_mutually_exclusive_group()
    init.add_argument("--backbone-state", help="Existing torchvision ResNet-18 state_dict.")
    init.add_argument("--init-sae", help="Earlier resnet18_sparse_ae.py checkpoint. Its dictionary dimensions take precedence.")


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="Fit the sparse encoder, decoder and latent classifier.")
    fitting_options(t)
    cv = commands.add_parser("crossval", help="Run independent stratified folds and summarize fixed-epoch accuracy.")
    fitting_options(cv, cv=True)
    cv.add_argument("--folds", type=int, default=10, help="Number of folds (default 10).")
    cv.add_argument("--fold", type=int, help="Run just one fold, numbered from 1 through --folds. Default runs all.")
    cv.add_argument("--resume", action="store_true", help="Continue matching partial folds and reuse completed histories.")
    r = commands.add_parser("resume", help="Continue from the end of a saved epoch.")
    runtime(r)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--epochs", type=int, required=True, help="Total epochs including those already completed.")
    r.add_argument("--data", help="Override the saved data location if the identical dataset moved.")

    e = commands.add_parser("evaluate", aliases=["eval"], help="Evaluate or sweep JPEG levels on identical held-out images.")
    runtime(e)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--limit", type=int, help="Defaults to the saved validation count. 0 uses all.")
    e.add_argument("--levels", type=jpeg_level, nargs="+", help="Example: --levels source 0 20 70 95")
    e.add_argument("--entire-input", action="store_true", help="Treat a separate single folder or CSV as a test set.")
    e.add_argument("--reconstruction", action="store_true",
                   help="Measure feature MSE and, for mapped classes, original-fc accuracy on reconstructed features. Uses last.pt.")
    e.add_argument("--baseline", action="store_true", help="Also load the original ImageNet fc for a standard best.pt.")
    e.add_argument("--json", type=Path, help="Write metrics for each level.")

    pred = commands.add_parser("predict", help="Classify one image and show its strongest sparse features.")
    enc = commands.add_parser("encode", help="Export sparse codes for an image or a folder to CSV.")
    for sub in (pred, enc):
        runtime(sub)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--compression-level", type=jpeg_level, default=argparse.SUPPRESS,
                         help="Override the saved level. Use source for an already-compressed image.")
    pred.add_argument("--image", type=Path, required=True)
    pred.add_argument("--top", type=int, default=5)
    pred.add_argument("--save-latent", type=Path, help="Also save the full 1-D latent tensor to a .pt file.")
    enc.add_argument("--input", type=Path, required=True)
    enc.add_argument("--output", type=Path, required=True)
    enc.add_argument("--batch-size", type=int, default=64)

    d = commands.add_parser("check", aliases=["demo"], help="Verify gradients, freezing and decoder-free inference with zero optimizer steps.")
    runtime(d)
    initialization_options(d)
    d.add_argument("--data", type=Path, help="Optional local labeled dataset. Default uses synthetic images.")
    d.add_argument("--pretrained", action="store_true", help="Check with ImageNet weights; may download them. Default is offline.")
    d.add_argument("--atoms", type=int, default=2048)
    d.add_argument("--k", type=int, default=32)
    d.add_argument("--decoder-hidden", type=int, default=0)
    d.add_argument("--batch-size", type=int, default=4)
    d.add_argument("--compression-level", type=jpeg_level, default=20)
    c = commands.add_parser("compress", help="Save a JPEG with the same encoder used by the model.")
    c.add_argument("input", type=Path)
    c.add_argument("output", type=Path, nargs="?")
    c.add_argument("--compression-level", type=int, default=20)
    x = commands.add_parser("convert", help="Convert an earlier linear SAE checkpoint into a decoder-free predictor.")
    x.add_argument("--sae", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--compression-level", type=jpeg_level, default=None)
    info = commands.add_parser("info", help="Show checkpoint settings, dimensions, metrics and optional training history.")
    info.add_argument("--checkpoint", type=Path, required=True)
    info.add_argument("--history", action="store_true")
    return p


def fold_history(cfg, directory, partitions, cache, args, device):
    checkpoint = directory / "last.pt"
    previous = None
    if args.resume and checkpoint.is_file():
        previous = read_checkpoint(checkpoint)
        if (previous["format"] != TRAINING or previous["protocol"].get("digest") != partitions[-1]
                or previous["classes"] != partitions[0].classes):
            raise ValueError(f"Data or class order differs from {checkpoint}. Start a new CV directory.")
        saved = Settings(**previous["settings"])
        ignored = {"data", "cache"}  # A moved dataset is checked by its split fingerprint.
        if cfg.init_sae:
            ignored.update(("atoms", "k", "decoder_hidden", "encoder_init"))
        changed = [f.name for f in fields(Settings) if f.name not in ignored
                   and getattr(saved, f.name) != getattr(cfg, f.name)]
        if changed:
            raise ValueError(f"Fold {cfg.fold} settings differ in {', '.join(changed)}. Choose a new --out.")
        if previous["epoch"] >= args.epochs:
            print(f"Reusing fold {cfg.fold}, already fitted through epoch {previous['epoch']}.", flush=True)
            return previous["history"]
        # An imported SAE can override the requested dimensions during the original fit.
        cfg = replace(saved, data=cfg.data, cache=cfg.cache)
    return train(cfg, directory, args.epochs, args.workers, device, previous,
                 partitions=partitions, fold_cache=cache)


def cv_summary(rows, folds, epoch):
    scores = [row["validation"]["top1"] for row in rows]
    total = sum(row["validation"]["images"] for row in rows)
    pooled = sum(row["validation"]["top1"] * row["validation"]["images"] for row in rows) / total
    return {"folds_completed": len(rows), "folds_planned": folds, "fixed_epoch": epoch,
            "validation_predictions": total, "pooled_top1": pooled, "mean_fold_top1": statistics.mean(scores),
            "sd_fold_top1": statistics.stdev(scores) if len(scores) > 1 else None, "results": rows}


def run_fitting(args, device):
    cv = args.command == "crossval"
    cfg = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)
                      if f.name not in ("fold", "folds") and hasattr(args, f.name)})
    cfg.validate()
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("Epochs must be positive and workers nonnegative.")
    if not cv and not args.plan_only:
        train(cfg, args.out, args.epochs, args.workers, device)
        return
    pair = sources(cfg.data)
    pool = pair[0]
    folds = args.folds if cv else 0
    if cv and (not 2 <= folds <= len(pool.labels) or (args.fold is not None and not 1 <= args.fold <= folds)):
        raise ValueError("Use 2 <= --folds <= image count and a --fold number from 1 through --folds.")
    if cv and pair[1] is not None:
        print("Cross-validation partitions the training split of this dataset.", flush=True)
    selected = range(1, folds + 1) if cv else (1,)
    if cv and args.fold is not None:
        selected = (args.fold,)
    cache = None
    if (len(selected) > 1 and not args.plan_only and cfg.train_limit == cfg.val_limit == 0
            and use_memory_cache(cfg.cache, len(pool.labels))):
        cache = FoldFeatureCache()
    results = []
    for number in selected:
        current = replace(cfg, folds=folds, fold=number)
        part = training_splits(cfg.data, cfg.train_limit, cfg.val_limit, cfg.holdout, cfg.seed,
                               folds, number, source_pair=pair)
        n_train, n_val = len(part[1]), len(part[3])
        steps = math.ceil(n_train / cfg.batch_size)
        plan = {"split": part[4], "training_images": n_train, "validation_images": n_val,
                "epochs": args.epochs, "batch_size": cfg.batch_size, "batches_per_epoch": steps,
                "planned_optimizer_steps": steps * args.epochs}
        if cv:
            plan.update(fold=number, folds=folds)
        print(json.dumps(plan), flush=True)
        if args.plan_only:
            continue
        directory = args.out / f"fold-{number:02d}"
        history = fold_history(current, directory, part, cache, args, device)
        # Use the prespecified epoch for CV scoring; best.pt selects on the outer fold.
        metrics = next(record["validation"] for record in history if record["epoch"] == args.epochs)
        results.append({"fold": number, "training_images": n_train, "validation": metrics})
        summary = cv_summary(results, folds, args.epochs)
        report = args.out / "cv_summary.json" if args.fold is None else directory / "fold_summary.json"
        with text_output(report) as f:
            json.dump(summary, f, indent=2, allow_nan=False)
    if results:
        print(f"Fixed-epoch top-1 over {summary['validation_predictions']:,} held-out predictions: "
              f"{100 * summary['pooled_top1']:.2f}%. Saved {report}", flush=True)


def main():
    p = cli()
    args = p.parse_args(sys.argv[1:] or ["check"])
    try:
        cmd = args.command
        if cmd == "compress":
            target = compress_jpeg(args.input, args.output, args.compression_level)
            print(f"{target}  ({args.input.stat().st_size:,} -> {target.stat().st_size:,} bytes)")
            return
        if cmd == "convert":
            export_legacy(args)
            return
        if cmd == "info":
            state = read_checkpoint(args.checkpoint)
            keys = ("format", "epoch", "atoms", "k", "compression_level", "protocol", "validation", "settings")
            details = {k: state[k] for k in keys if k in state}
            details["classes"] = len(state["classes"])
            details["fc_weight_shape"] = list(state["model"]["fc.weight"].shape)
            if args.history:
                details["history"] = state.get("history", [])
            print(json.dumps(details, indent=2))
            return
        device = torch.device("cpu") if getattr(args, "plan_only", False) else device_for(args)
        if getattr(args, "batch_size", 1) < 1:
            raise ValueError("--batch-size must be positive.")
        if cmd in ("train", "crossval"):
            lock_all = cmd == "crossval" and args.fold is None and not args.plan_only
            with run_writer(args.out) if lock_all else nullcontext():
                run_fitting(args, device)
        elif cmd == "resume":
            state = read_checkpoint(args.checkpoint)
            if state["format"] != TRAINING:
                raise ValueError("Use last.pt to resume training.")
            cfg = Settings(**state["settings"])
            cfg.data = args.data or cfg.data
            train(cfg, args.checkpoint.parent, args.epochs, args.workers, device, resume=state)
        elif cmd in ("evaluate", "eval"):
            run_evaluation(args, device)
        elif cmd == "predict":
            predict(args, device)
        elif cmd == "encode":
            encode(args, device)
        elif cmd in ("check", "demo"):
            check_pipeline(args, device)
    except (ValueError, OSError) as exc:
        p.exit(2, f"{p.prog}: {exc}\n")


if __name__ == "__main__":
    main()
