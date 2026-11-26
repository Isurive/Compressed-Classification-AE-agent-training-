# ENG-002: PyTorch environment, `main.py check`, and laptop throughput

**Agent:** `claude-orient-20261122-a`. **Role:** engineer. **Date:** 2026-11-23.
**Status:** Submitted for review. The user approved the PyTorch install ("Yes, install it") and asked for the `--pretrained` check. `src/` was not edited. No dataset was downloaded.

## Environment (measured)

| Item | Value |
| --- | --- |
| Conda env | `ai360` at `C:\Users\ericw\miniconda3\envs\ai360`. Created with `conda create -n ai360 -c conda-forge --override-channels python=3.12 pip` (conda-forge only, so no Anaconda-channel terms were needed). |
| Install | `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Versions | Python 3.12.16, torch 2.11.0+cu128, torchvision 0.26.0+cu128, Pillow 13.0.0, numpy 2.5.4, cuDNN 91900 |
| GPU | `torch.cuda.is_available()` True. NVIDIA GeForce RTX 3080 Ti Laptop GPU (16 GB), driver 610.60. A matmul smoke test passed. |
| Run with | `C:\Users\ericw\miniconda3\envs\ai360\python.exe`, or `conda activate ai360`, then `python ...` |

Note: Pillow 13.0 in `ai360` differs from Pillow 12.1 in `py` (Python 3.14). JPEG byte counts depend on the encoder version, so record the env used for every byte measurement. ENG-001's manifest already records the Pillow version.

## `main.py check` (verbatim outputs saved)

The command follows the user's notebook recommendation: `python -B main.py check --batch-size 4 --atoms 2048 --k 32 --encoder-init he --fc-init xavier --device <dev> [--pretrained]`, run from `src/`.

| Run | File | Result |
| --- | --- | --- |
| CPU, random backbone | [check_cpu.txt](check_cpu.txt) | PASS |
| CUDA, random backbone | [check_cuda.txt](check_cuda.txt) | PASS |
| CUDA, `--pretrained` | [check_cuda_pretrained.txt](check_cuda_pretrained.txt) | PASS. Downloaded `resnet18-f37072fd.pth` (46.8 MB) to `C:\Users\ericw\.cache\torch\hub\checkpoints\` |

Each PASS confirms all of the following: finite, nonzero gradients for the encoder, decoder, and latent fc; no gradient to the backbone; all weights and BatchNorm buffers unchanged (zero optimizer steps); at most 32 active atoms; no decoder weights in the deployment state; and identical outputs from the decoder-free reload. The checks used synthetic images, so this verifies the code paths, not the accuracy. The trailing `bc: command not found` / `wall s` line in each file is a broken timing wrapper in my shell, not a failure. Each run exited with code 0.

## Throughput on this laptop (synthetic inputs, [bench.py](bench.py), [bench_output.json](bench_output.json))

| Stage | Hardware | Measured |
| --- | --- | --- |
| JPEG round trip (level 20) + ResNet preprocessing, about 500×375 image | CPU, 1 process | 7.7 ms/image, about 129 images/s |
| Frozen pretrained ResNet-18 features, batch 128, 224², fp32 | GPU | about 2,660 images/s, 0.9 GB peak |
| Sparse-head training step (M=2048, k=32, CE + MSE + dependency + coherence), batch 128 | GPU | about 127 steps/s, 0.2 GB peak |

**Estimates for the README's 45k/5k ImageNet-val run [inference, not measured on real data]:**
- **Bottleneck:** CPU JPEG and preprocessing, not the GPU. One feature-extraction pass over 50k images takes about 6–7 min with `--workers 0` (the default), and probably 1.5–3 min with `--workers 4–8`. The i9-12900H has 20 logical CPUs. Real ImageNet decoding and disk reads may be slower than these synthetic numbers.
- **Training on cached features:** 352 steps/epoch at about 127 steps/s is about 3 s/epoch, so 10 epochs take under a minute. 10-fold CV reuses one feature cache: about 5 min of head training plus one extraction pass.
- **A grid:** for example, 5 compression levels × 3 seeds × (dictionary + baseline head) is roughly 1–2 hours in total, dominated by one extraction pass per level.
- **Memory:** cached features are about 98 MiB. GPU use stays under about 1 GB, so the 16 GB GPU and 32 GB RAM have ample headroom.

**Conclusion:** with the frozen backbone, this laptop is sufficient for the planned experiments. It would not be sufficient for training a ResNet from scratch on ImageNet, which the current design does not need.

## Practical recommendations

- Pass `--cache memory` for 45k images (`auto` streams above 20k) and `--workers 4` to `8`.
- Keep ImageNet data **outside the OneDrive folder**. A 6+ GB dataset would sync to the cloud and slow disk reads. Keeping `runs/` outside OneDrive is also reasonable.
- Run on AC power. Laptop GPUs throttle on battery and under heat; this matters little given the run times above.

## Unexecuted

- Not tested on real ImageNet files.
- `--workers > 0` on Windows was not tested.
- The wall-clock estimates above are extrapolations from synthetic benchmarks.
