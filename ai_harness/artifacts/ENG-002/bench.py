"""Rough throughput benchmark for this laptop (synthetic data; no dataset needed).

    C:/Users/ericw/miniconda3/envs/ai360/python.exe -B ai_harness/artifacts/ENG-002/bench.py

Measures the three costs of src/main.py training:
  1. CPU: JPEG round trip + ResNet preprocessing per image (single process).
  2. GPU: frozen pretrained ResNet-18 feature extraction (batch 128, 224x224, fp32).
  3. GPU: one sparse-head training step on cached 512-d features (M=2048, k=32, batch 128),
     using the same loss terms as sparse_workflow.training_pass.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from compress_jpeg import model_image  # noqa: E402
from resnet18_sparse_ae import SparseDictionaryAE, sparse_codes  # noqa: E402

results = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}

# 1. CPU preprocessing on ImageNet-sized synthetic photos (about 500x375, typical of ImageNet val).
rng = np.random.default_rng(0)
y, x = np.mgrid[0:375, 0:500] / 500
photos = []
for _ in range(40):
    base = np.stack([x, y, 1 - x * y], -1) * 200 + rng.normal(0, 12, (375, 500, 3))
    photos.append(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)))
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
for img in photos[:5]:
    transform(model_image(img, 20))  # warm up
start = time.perf_counter()
for img in photos:
    transform(model_image(img, 20))
per_image = (time.perf_counter() - start) / len(photos)
results["cpu_preprocess_ms_per_image_one_process"] = round(1000 * per_image, 2)
results["cpu_preprocess_images_per_s_one_process"] = round(1 / per_image, 1)

# 2. GPU frozen ResNet-18 features.
dev = torch.device("cuda")
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(dev).eval()
backbone.fc = torch.nn.Identity()
batch = torch.randn(128, 3, 224, 224, device=dev)
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for _ in range(5):
        backbone(batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        backbone(batch)
    torch.cuda.synchronize()
results["gpu_feature_images_per_s"] = round(20 * 128 / (time.perf_counter() - start))
results["gpu_feature_peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)

# 3. GPU sparse-head training step on cached features.
sae = SparseDictionaryAE(512, 2048, 32, 0, "he").to(dev)
fc = torch.nn.Linear(2048, 1000).to(dev)
opt = torch.optim.Adam([*sae.parameters(), *fc.parameters()], lr=1e-3)
feats = torch.randn(128, 512, device=dev).relu()
labels = torch.randint(1000, (128,), device=dev)


def step():
    opt.zero_grad(set_to_none=True)
    z = sparse_codes(feats, sae.center, sae.encoder, sae.k)
    loss = (F.cross_entropy(fc(z), labels) + F.mse_loss(sae.decode(z), feats)
            + 0.1 * sae.dependency_loss(z) + 0.05 * sae.coherence_loss())
    loss.backward()
    opt.step()


torch.cuda.reset_peak_memory_stats()
for _ in range(10):
    step()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(200):
    step()
torch.cuda.synchronize()
results["gpu_head_steps_per_s_batch128"] = round(200 / (time.perf_counter() - start), 1)
results["gpu_head_peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
print(json.dumps(results, indent=2))
