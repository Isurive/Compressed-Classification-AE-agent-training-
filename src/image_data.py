"""Local ImageNet inputs, repeatable class-balanced subsets and JPEG preprocessing."""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import random
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.models import ResNet18_Weights

from compress_jpeg import model_image


class ImageSource:
    def __init__(self, root, classes, labels, paths=None, hf=None):
        self.root, self.classes, self.labels = Path(root), list(classes), list(labels)
        self.paths, self.hf = paths, hf

    def key(self, i):
        if self.paths is None:
            return f"{self.hf._fingerprint}/{i}"
        return self.paths[i].as_posix()

    def image(self, i):
        if self.paths is not None:
            with Image.open(self.root / self.paths[i]) as image:
                return image.copy()
        value = self.hf[i]["image"]
        if isinstance(value, Image.Image):
            return value
        content = value.get("bytes")
        with Image.open(BytesIO(content) if content is not None else value["path"]) as image:
            return image.copy()

    def align(self, classes, saved_ids=None):
        """Remap local indices into the classifier's order, even for a partial evaluation set."""
        if self.classes == classes:
            return
        index = {name: i for i, name in enumerate(classes)}
        unknown = set(self.classes) - set(classes)
        if unknown:
            actual_ids = reference_ids(self.classes, "auto")
            if saved_ids is None or actual_ids is None or not set(actual_ids).issubset(saved_ids):
                raise ValueError(f"Unknown evaluation classes: {sorted(unknown)[:5]}")
            canonical = {n: i for i, n in enumerate(saved_ids)}
            remap = [canonical[n] for n in actual_ids]
        else:
            remap = [index[name] for name in self.classes]
        self.labels = [remap[i] for i in self.labels]
        self.classes = list(classes)


def class_order(names):
    names = set(names)
    return sorted(names, key=int) if all(name.isdecimal() for name in names) else sorted(names)


def folder_source(root):
    folder = ImageFolder(root)
    classes = class_order(folder.classes)
    remap = {i: classes.index(name) for i, name in enumerate(folder.classes)}
    paths = [Path(path).relative_to(root) for path, _ in folder.samples]
    labels = [remap[y] for y in folder.targets]
    return ImageSource(root, classes, labels, paths=paths)


def official_flat_val(root, files):
    classes = Path(__file__).with_name("imagenet1k_wnids.txt").read_text().splitlines()
    index = {name: i for i, name in enumerate(classes)}
    with gzip.open(Path(__file__).with_name("imagenet1k_val_labels.txt.gz"), "rt") as f:
        labels = [index[line.strip()] for line in f]
    paths, targets, seen = [], [], set()
    for p in sorted(files):
        match = re.fullmatch(r"ILSVRC2012_val_(\d{8})", p.stem)
        number = int(match[1]) if match else 0
        if not 1 <= number <= len(labels) or number in seen:
            raise ValueError(f"Invalid or duplicate ILSVRC2012 validation image ID in {p.name}.")
        seen.add(number)
        paths.append(p.relative_to(root))
        targets.append(labels[number - 1])
    return ImageSource(root, classes, targets, paths=paths)


def csv_source(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        image_key = next((k for k in ("path", "filename", "file_name", "image") if k in fields), None)
        label_key = next((k for k in ("label", "class_id", "class", "wnid") if k in fields), None)
        if not image_key or not label_key:
            raise ValueError("A CSV manifest needs path and label columns. Paths are relative to the CSV.")
        paths, names, seen = [], [], set()
        for row in reader:
            raw_path, label = row[image_key].strip(), row[label_key].strip()
            if not raw_path or not label:
                raise ValueError(f"Missing path or label on CSV line {reader.line_num}.")
            image_path = Path(raw_path)
            key = os.path.normcase(os.path.abspath(path.parent / image_path))
            if key in seen:
                raise ValueError(f"Repeated image path in the CSV manifest: {raw_path}")
            seen.add(key)
            paths.append(image_path)
            names.append(label)
    classes = class_order(names)
    index = {name: i for i, name in enumerate(classes)}
    return ImageSource(path.parent, classes, [index[n] for n in names], paths=paths)


def hf_source(root, data):
    if not {"image", "label"}.issubset(data.column_names):
        raise ValueError("A saved Hugging Face Dataset needs image and label columns.")
    feature = data.features["label"]
    labels = list(data["label"])
    if getattr(feature, "names", None):
        classes = list(feature.names)
        if any(y < 0 or y >= len(classes) for y in labels):
            raise ValueError("Training and evaluation require labeled rows.")
    else:
        names = [str(value) for value in labels]
        classes = class_order(names)
        index = {name: i for i, name in enumerate(classes)}
        labels = [index[name] for name in names]
    return ImageSource(root, classes, labels, hf=data)


def sources(path, evaluation_only=False):
    """Return (training source, optional distinct validation source)."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        return csv_source(path), None
    if not path.is_dir():
        raise ValueError(f"Dataset path does not exist: {path}")
    if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
        try:
            from datasets import DatasetDict, load_from_disk
        except ImportError as exc:
            raise ValueError("Install the optional reader with: python -m pip install datasets") from exc
        data = load_from_disk(str(path))
        if not isinstance(data, DatasetDict):
            return hf_source(path, data), None
        valid_key = next((k for k in ("validation", "val", "valid") if k in data), None)
        if "train" in data and valid_key:
            if evaluation_only:
                return None, hf_source(path, data[valid_key])
            return hf_source(path, data["train"]), hf_source(path, data[valid_key])
        key = valid_key or ("train" if "train" in data else None)
        if key is None:
            raise ValueError("Expected a train or validation split in the saved DatasetDict.")
        return hf_source(path, data[key]), None
    val = next((path / k for k in ("val", "validation", "valid") if (path / k).is_dir()), None)
    train = path / "train"
    if train.is_dir() and val is not None:
        if evaluation_only:
            return None, sources(val)[0]
        return sources(train)[0], sources(val)[0]
    if val is not None or train.is_dir():
        return sources(val or train)
    for name in ("metadata.csv", "labels.csv", "manifest.csv"):
        manifest = path / name
        if manifest.is_file():
            with manifest.open(encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f), [])
            if any(k in header for k in ("path", "filename", "file_name", "image")):
                return csv_source(manifest), None
    flat = [p for p in path.glob("ILSVRC2012_val_*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
    if flat:
        return official_flat_val(path, flat), None
    return folder_source(path), None


def group_indices(indices, labels):
    groups = defaultdict(list)
    for i in indices:
        groups[labels[i]].append(i)
    return groups


def holdout_indices(labels, fraction, seed):
    rng = random.Random(seed)
    train, val = [], []
    for group in group_indices(range(len(labels)), labels).values():
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * fraction)))
        val.extend(group[:count])
        train.extend(group[count:])
    if not val:
        raise ValueError("A holdout split needs at least two images in at least one class.")
    return train, val


def fold_indices(labels, folds, fold, seed):
    """Deterministic, disjoint, one-based stratified folds, balanced even for uneven classes."""
    if not 2 <= folds <= len(labels) or not 1 <= fold <= folds:
        raise ValueError("Use 2 <= folds <= image count and 1 <= fold <= folds.")
    rng = random.Random(seed)
    train, val = [], []
    offset = 0
    for group in group_indices(range(len(labels)), labels).values():
        rng.shuffle(group)
        for position, i in enumerate(group):
            if (offset + position) % folds == fold - 1:
                val.append(i)
            else:
                train.append(i)
        # Rotate remainder assignments so the overall fold sizes also differ by at most one.
        offset = (offset + len(group)) % folds
    return train, val


def balanced_indices(indices, labels, limit, seed):
    if limit == 0 or limit >= len(indices):
        return list(indices)
    rng = random.Random(seed)
    queues = list(group_indices(indices, labels).values())
    rng.shuffle(queues)
    for group in queues:
        rng.shuffle(group)
    chosen = []
    while len(chosen) < limit:
        remaining = []
        for group in queues:
            chosen.append(group.pop())
            if len(chosen) == limit:
                return chosen
            if group:
                remaining.append(group)
        queues = remaining
    return chosen


def training_splits(path, train_limit, val_limit, holdout, seed, folds=0, fold=1, source_pair=None):
    train, val = sources(path) if source_pair is None else source_pair
    if folds:
        train_ids, val_ids = fold_indices(train.labels, folds, fold, seed)
        val, split = train, "stratified k-fold"
    elif val is None:
        train_ids, val_ids = holdout_indices(train.labels, holdout, seed)
        val, split = train, "single-pool holdout"
    else:
        val.align(train.classes, reference_ids(train.classes, "auto"))
        train_ids, val_ids = range(len(train.labels)), range(len(val.labels))
        split = "separate train/validation"
    train_ids = balanced_indices(train_ids, train.labels, train_limit, seed + 1)
    val_ids = balanced_indices(val_ids, val.labels, val_limit, seed + 2)
    if not train_ids or not val_ids or len(train.classes) < 2:
        raise ValueError("Need labeled training and validation images and at least two classes.")
    digest = hashlib.sha256()
    for tag, source, ids in (("train", train, train_ids), ("val", val, val_ids)):
        digest.update(tag.encode())
        for i in ids:
            digest.update(f"{source.key(i)}\t{source.labels[i]}\n".encode())
    return train, train_ids, val, val_ids, split, digest.hexdigest()


def evaluation_split(path, classes, config, split, limit, entire, saved_ids=None):
    use_fold = bool(config.get("folds")) and not entire
    first, second = sources(path, evaluation_only=not use_fold)
    source = first if use_fold else (second or first)
    source.align(classes, saved_ids)
    indices = range(len(source.labels))
    if use_fold:
        _, indices = fold_indices(source.labels, config["folds"], config["fold"], config["seed"])
    elif second is None and split == "single-pool holdout" and not entire:
        _, indices = holdout_indices(source.labels, config["holdout"], config["seed"])
    return source, balanced_indices(indices, source.labels, limit, config["seed"] + 2)


def reference_ids(classes, label_space):
    """Resolve each local class to the corresponding original ImageNet classifier row."""
    if label_space == "custom":
        return None
    if label_space == "imagenet-order":
        if len(classes) != 1000:
            raise ValueError("imagenet-order requires 1,000 labels in canonical order.")
        return list(range(1000))
    wnids = Path(__file__).with_name("imagenet1k_wnids.txt").read_text().splitlines()
    lookup = {wnid: i for i, wnid in enumerate(wnids)}
    categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    for i, name in enumerate(categories):
        lookup.setdefault(name.casefold(), i)
    ids = []
    for name in classes:
        key = name.casefold()
        wnid = re.search(r"(?<![a-z0-9])n[0-9]{8}(?![0-9])", key)
        number = re.match(r"^(\d+)(?:$|[_ -])", key)
        if wnid and wnid[0] in lookup:
            ids.append(lookup[wnid[0]])
        elif number:
            ids.append(int(number[1]))
        elif key in lookup:
            ids.append(lookup[key])
        else:
            # HF ImageNet-1K exposes ordered WordNet synonym lists as ClassLabel names.
            if len(classes) == 1000 and [n.split(",")[0].lower() for n in classes[:3]] == [
                "tench", "goldfish", "great white shark"
            ] and classes[-1].lower().split(",")[0] == "toilet tissue":
                return list(range(1000))
            return None
    return ids if len(set(ids)) == len(ids) and all(0 <= i < 1000 for i in ids) else None


class ImageView(Dataset):
    def __init__(self, source, indices, level):
        self.source, self.indices, self.level = source, indices, level
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, row):
        i = self.indices[row]
        try:
            with self.source.image(i) as image:
                tensor = self.transform(model_image(image, self.level))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot process {self.source.key(i)}: {exc}") from exc
        return tensor, self.source.labels[i]


class UnlabeledImages(Dataset):
    def __init__(self, path, level):
        path = Path(path)
        self.root = path.parent if path.is_file() else path
        self.paths = ([path] if path.is_file() else
                      sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS))
        if not self.paths:
            raise ValueError(f"No images found at {path}")
        self.level = level
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, row):
        path = self.paths[row]
        with Image.open(path) as image:
            tensor = self.transform(model_image(image, self.level))
        return tensor, path.relative_to(self.root).as_posix()
