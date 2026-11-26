"""Sparse autoencoding and diverse dictionary learning on TorchVision ResNet-18 features.

Train on an ImageFolder copy of ImageNet-1K with train/<WNID>/ and val/<WNID>/.
    python resnet18_sparse_ae.py --data /datasets/imagenet --output resnet18_sae.pt

Attach the head to a model whose weights you already loaded.
    model = ResNet18SparseAE(existing_resnet18, n_atoms=2048, k=32)
    model.initialize_center(images)  # once, on a representative training batch
    out = model(images)  # images use the backbone's normal preprocessing
    loss = (F.mse_loss(out.reconstruction, out.target)
            + 0.1 * model.sae.dependency_loss(out.codes)
            + 0.05 * model.sae.coherence_loss())

Load the best trained model for inference.
    model = load_trained("resnet18_sae.pt")
    with torch.inference_mode():
        out = model(images)
        predicted_classes = out.logits.argmax(dim=1)

The decoder dependency penalty adapts Zheng et al., Diverse Dictionary Learning,
https://arxiv.org/abs/2604.17568. Top-K controls per-image activation sparsity.
Requires Python 3.10+ and matching torch / torchvision packages (torch 2.2+).
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet, ResNet18_Weights, resnet18


def sparse_codes(features: Tensor, center: Tensor, encoder: nn.Linear, k: int) -> Tensor:
    """The shared sparse encoder used by both the training and deployed models."""
    scores = encoder(features - center)
    values, indices = scores.topk(k, dim=-1, sorted=False)
    return torch.zeros_like(scores).scatter(-1, indices, values.relu())


class SparseDictionaryAE(nn.Module):
    """Top-K encoder, unit-norm dictionary and optional nonlinear decoder residual."""

    def __init__(self, feature_dim: int = 512, n_atoms: int = 2048, k: int = 32,
                 decoder_hidden: int = 0, encoder_init: str = "unit"):
        super().__init__()
        if feature_dim < 1 or n_atoms < 2 or not 1 <= k <= n_atoms:
            raise ValueError("Require feature_dim >= 1, n_atoms >= 2 and 1 <= k <= n_atoms.")
        if decoder_hidden < 0:
            raise ValueError("decoder_hidden must be nonnegative.")
        self.k = k
        self.decoder_hidden = decoder_hidden
        self.encoder = nn.Linear(feature_dim, n_atoms)
        self.center = nn.Parameter(torch.zeros(feature_dim))
        if encoder_init == "he":
            nn.init.kaiming_normal_(self.encoder.weight, mode="fan_in", nonlinearity="relu")
            atoms = F.normalize(self.encoder.weight.detach(), dim=1)
        elif encoder_init == "unit":
            atoms = F.normalize(torch.randn(n_atoms, feature_dim), dim=1)
            with torch.no_grad():
                self.encoder.weight.copy_(atoms)
        else:
            raise ValueError("encoder_init must be 'unit' or 'he'.")
        # Both schemes align the directions at startup. Parameters then learn independently.
        self.atom_weights = nn.Parameter(atoms)
        nn.init.zeros_(self.encoder.bias)
        self.residual_decoder = None
        if decoder_hidden:
            self.residual_decoder = nn.Sequential(nn.Linear(n_atoms, decoder_hidden), nn.Tanh(),
                                                  nn.Linear(decoder_hidden, feature_dim, bias=False))
            nn.init.zeros_(self.residual_decoder[-1].weight)  # Start with the linear dictionary.

    @property
    def dictionary(self) -> Tensor:
        """The linear decoder's unit-length atoms, stored as [n_atoms, feature_dim]."""
        return F.normalize(self.atom_weights, dim=1)

    def encode(self, features: Tensor) -> Tensor:
        return sparse_codes(features, self.center, self.encoder, self.k)

    def decode(self, codes: Tensor) -> Tensor:
        reconstruction = codes @ self.dictionary + self.center
        if self.residual_decoder is not None:
            reconstruction = reconstruction + self.residual_decoder(codes)
        return reconstruction

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(features)
        return codes, self.decode(codes)

    def dependency_loss(self, codes: Tensor) -> Tensor:
        """Scale-invariant L1 proxy for sparsity of the decoder's latent-to-channel Jacobian."""
        if self.residual_decoder is None:
            # The Jacobian of the linear decoder is exactly dictionary.T for every image.
            directions = self.dictionary
        else:
            # One uniformly sampled Jacobian column per image, including inactive latents.
            # JVP returns the selected latent's effect on all output channels. Detaching
            # the evaluation points confines this regularizer's gradients to the decoder.
            points = codes.detach().reshape(-1, codes.shape[-1])
            ids = torch.randint(points.shape[1], (points.shape[0], 1), device=points.device)
            tangent = torch.zeros_like(points).scatter_(1, ids, 1.0)
            _, columns = torch.func.jvp(self.decode, (points,), (tangent,))
            directions = F.normalize(columns, dim=-1)
        # Equivalent to ||column||_1 / (sqrt(feature_dim) * ||column||_2) averaged
        # over columns. Concentrating a direction on fewer output channels reduces it.
        return directions.abs().mean() * self.encoder.in_features ** 0.5

    def coherence_loss(self, n_anchors: int | None = 256) -> Tensor:
        """Mean worst squared cosine with another atom. Pass None to score every atom."""
        atoms = self.dictionary
        n_atoms = atoms.shape[0]
        if n_anchors is None or n_anchors >= n_atoms:
            anchors = torch.arange(n_atoms, device=atoms.device)
        elif n_anchors >= 1:
            anchors = torch.randperm(n_atoms, device=atoms.device)[:n_anchors]
        else:
            raise ValueError("n_anchors must be positive or None.")
        # Compare each sampled atom to every other atom in the dictionary.
        cosine_sq = (atoms[anchors] @ atoms.T).square()
        cosine_sq = cosine_sq.scatter(1, anchors[:, None], 0.0)
        return cosine_sq.amax(dim=1).mean()


class SAEOutput(NamedTuple):
    features: Tensor        # [B, 512], input to the sparse encoder
    target: Tensor          # [B, 512], original pretrained pre-fc features
    codes: Tensor           # [B, n_atoms], at most k positive entries per image
    reconstruction: Tensor  # [B, 512], decoded feature vector
    logits: Tensor          # [B, 1000] for ImageNet-1K, classifier applied to reconstruction


class ResNet18SparseAE(nn.Module):
    """Attach an SAE to a copy of an existing TorchVision ResNet-18."""

    def __init__(self, backbone: ResNet, n_atoms: int = 2048, k: int = 32,
                 finetune_layer4: bool = False, decoder_hidden: int = 0, encoder_init: str = "unit"):
        super().__init__()
        if not isinstance(backbone.fc, nn.Linear) or backbone.fc.in_features != 512:
            raise ValueError("Expected a TorchVision ResNet-18 with its 512-input fc layer.")
        self.backbone = copy.deepcopy(backbone).requires_grad_(False)
        self.tuned_layer4 = (
            copy.deepcopy(self.backbone.layer4).requires_grad_(True) if finetune_layer4 else None
        )
        self.sae = SparseDictionaryAE(backbone.fc.in_features, n_atoms, k,
                                      decoder_hidden, encoder_init).to(backbone.fc.weight)
        self.train()

    def train(self, mode: bool = True) -> ResNet18SparseAE:
        super().train(mode)
        self.backbone.eval()  # Freeze BatchNorm running statistics as well as parameters.
        return self

    @torch.no_grad()
    def _reference_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(images))))
        x = b.layer1(x)
        x = b.layer2(x)
        shared = b.layer3(x)
        target = b.avgpool(b.layer4(shared)).flatten(1)
        return shared, target

    @torch.no_grad()
    def initialize_center(self, images: Tensor) -> None:
        """Initialize the trainable feature offset once from a training batch."""
        _, targets = self._reference_features(images)
        self.sae.center.copy_(targets.mean(dim=0))

    def forward(self, images: Tensor) -> SAEOutput:
        shared, target = self._reference_features(images)
        features = target
        if self.tuned_layer4 is not None:
            features = self.backbone.avgpool(self.tuned_layer4(shared)).flatten(1)
        codes, reconstruction = self.sae(features)
        # A frozen fc still passes classification gradients back to the reconstruction.
        return SAEOutput(features, target, codes, reconstruction, self.backbone.fc(reconstruction))


def train_epoch(model: ResNet18SparseAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int, dependency_weight: float = 0.1,
                coherence_weight: float = 0.05, ce_weight: float = 0.0) -> None:
    model.train()
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        mse = F.mse_loss(out.reconstruction, out.target)
        dep = model.sae.dependency_loss(out.codes) if dependency_weight else mse.new_zeros(())
        coherence = model.sae.coherence_loss() if coherence_weight else mse.new_zeros(())
        loss = mse + dependency_weight * dep + coherence_weight * coherence
        if ce_weight:
            labels = labels.to(device, non_blocking=True)
            loss = loss + ce_weight * F.cross_entropy(out.logits, labels)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == len(loader):
            print(f"epoch {epoch}  step {step}/{len(loader)}  loss {loss.item():.5f}  "
                  f"feature MSE {mse.item():.5f}  dependency {dep.item():.5f}  "
                  f"coherence {coherence.item():.5f}", flush=True)


@torch.inference_mode()
def evaluate(model: ResNet18SparseAE, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compare the reconstructed and original features through the same classifier."""
    model.eval()
    n_images = 0
    total_mse = 0.0
    original_correct = reconstructed_correct = same_prediction = 0
    firing_counts = torch.zeros(model.sae.encoder.out_features, dtype=torch.long, device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        original = model.backbone.fc(out.target).argmax(dim=1)
        reconstructed = out.logits.argmax(dim=1)
        n_images += labels.numel()
        total_mse += F.mse_loss(out.reconstruction, out.target).item() * labels.numel()
        original_correct += (original == labels).sum().item()
        reconstructed_correct += (reconstructed == labels).sum().item()
        same_prediction += (original == reconstructed).sum().item()
        firing_counts += (out.codes > 0).sum(dim=0)
    if n_images == 0:
        raise ValueError("The validation loader yielded no images.")
    return {
        "feature_mse": total_mse / n_images,
        "original_top1": original_correct / n_images,
        "reconstructed_top1": reconstructed_correct / n_images,
        "top1_agreement": same_prediction / n_images,
        "active_per_image": firing_counts.sum().item() / n_images,
        "used_atom_fraction": (firing_counts > 0).float().mean().item(),
    }


def save_trained(model: ResNet18SparseAE, path: str | Path) -> None:
    """Save the complete model so inference can run offline, including a tuned layer4."""
    checkpoint = {
        "num_classes": model.backbone.fc.out_features,
        "n_atoms": model.sae.encoder.out_features,
        "k": model.sae.k,
        "finetune_layer4": model.tuned_layer4 is not None,
        "decoder_hidden": model.sae.decoder_hidden,
        "model": model.state_dict(),
    }
    atomic_torch_save(checkpoint, path)


def atomic_torch_save(checkpoint: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_trained(path: str | Path, device: str | torch.device = "cpu") -> ResNet18SparseAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    backbone = resnet18(weights=None, num_classes=checkpoint["num_classes"])
    model = ResNet18SparseAE(backbone, checkpoint["n_atoms"], checkpoint["k"],
                             checkpoint["finetune_layer4"], checkpoint["decoder_hidden"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="ImageNet-1K root with train/ and val/ class folders.")
    parser.add_argument("--backbone-state", type=Path, help="Optional plain torchvision ResNet-18 state_dict.")
    parser.add_argument("--output", type=Path, default=Path("resnet18_sae.pt"), help="Best validation feature-MSE checkpoint.")
    parser.add_argument("--atoms", type=int, default=2048)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--encoder-init", choices=("unit", "he"), default="unit")
    parser.add_argument("--dependency-weight", type=float, default=0.1, help="Decoder Jacobian sparsity weight.")
    parser.add_argument("--coherence-weight", type=float, default=0.05, help="Penalty on high absolute cosine between linear atoms.")
    parser.add_argument("--decoder-hidden", type=int, default=0, help="Nonlinear decoder residual width. 0 uses a linear dictionary.")
    parser.add_argument("--ce-weight", type=float, default=0.0, help="Optional ImageNet cross-entropy weight.")
    parser.add_argument("--finetune-layer4", action="store_true", help="Train a copy of layer4 against the original features.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Sparse head learning rate.")
    parser.add_argument("--layer4-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive, and workers must be nonnegative.")
    if min(args.dependency_weight, args.coherence_weight, args.ce_weight) < 0 or min(args.lr, args.layer4_lr) <= 0:
        parser.error("Loss weights must be nonnegative and learning rates must be positive.")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    weights = ResNet18_Weights.IMAGENET1K_V1
    # Match the feature distribution seen at inference, including resize, crop and normalization.
    transform = weights.transforms()
    train_set = ImageFolder(args.data / "train", transform=transform)
    val_set = ImageFolder(args.data / "val", transform=transform)
    if len(train_set.classes) != 1000 or train_set.class_to_idx != val_set.class_to_idx:
        parser.error("Expected the same 1,000 ILSVRC2012 class folders in train/ and val/.")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    backbone = resnet18(weights=weights if args.backbone_state is None else None)
    if args.backbone_state is not None:
        backbone.load_state_dict(torch.load(args.backbone_state, map_location="cpu", weights_only=True))
    model = ResNet18SparseAE(backbone, args.atoms, args.k, args.finetune_layer4,
                             args.decoder_hidden, args.encoder_init).to(device)
    del backbone
    # A batch estimate of the mean reduces the large shared offset of pooled ReLU features.
    model.initialize_center(next(iter(train_loader))[0].to(device))
    groups = [{"params": model.sae.parameters(), "lr": args.lr}]
    if model.tuned_layer4 is not None:
        groups.append({"params": model.tuned_layer4.parameters(), "lr": args.layer4_lr})
    optimizer = torch.optim.Adam(groups)
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch, args.dependency_weight,
                    args.coherence_weight, args.ce_weight)
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch}  validation  " + "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()),
              flush=True)
        if metrics["feature_mse"] < best_mse:
            best_mse = metrics["feature_mse"]
            save_trained(model, args.output)
            print(f"Saved {args.output} (validation feature MSE {best_mse:.5f})", flush=True)


if __name__ == "__main__":
    main()
