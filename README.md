# ImageClassification

**CIFAR‑10 image classification with a ResNet‑18 backbone (PyTorch)**

---

## Project summary

This repository demonstrates a clean, notebook‑friendly image classification training loop using PyTorch and `torchvision`. The notebook trains a ResNet‑18 (adapted for 32×32 CIFAR‑10 images) on the CIFAR‑10 dataset with common data augmentation, standard training utilities (checkpointing, evaluation), and a cosine annealing learning‑rate schedule.

## What the notebook does

* Downloads and prepares CIFAR‑10 (`torchvision.datasets.CIFAR10`) with simple but effective transforms:

  * training: random crop + horizontal flip + normalize
  * test: normalize
* Builds a ResNet‑18 backbone (optionally with ImageNet weights) and adapts it for 32×32 inputs by:

  * replacing the first conv to `kernel_size=3, stride=1, padding=1`
  * removing the initial maxpool (`nn.Identity()`)
  * replacing the final `fc` layer with `nn.Linear(..., num_classes)`
* Uses `CrossEntropyLoss`, SGD (`momentum=0.9`, `weight_decay=5e-4`) and `CosineAnnealingLR` scheduler.
* Implements notebook‑friendly training/evaluation loops with progress bars (tqdm), checkpoint saving (`last.pth`, `best.pth`, `interrupt.pth`), and quick visualizations of batches and model predictions.

## Default configuration (top of the notebook)

* `DATA_DIR = ./data`
* `CKPT_DIR = ./ckpt`
* `LOG_DIR = ./runs/exp1` (TensorBoard `SummaryWriter` is imported and ready)
* `BATCH_SIZE = 128`
* `EPOCHS = 30`
* `LR = 0.1`
* `NUM_WORKERS = None` (the notebook chooses `num_workers=0` on Windows and a small safe value on other OSes)
* `SEED = 42`
* `USE_PRETRAINED = False` (set to `True` to load ImageNet weights)
* `RESUME_FROM = None` (path to a checkpoint to resume training)

## Example results (from a sample run in the notebook)

The example training run included in the notebook reached the following validation accuracy/progression:

* Early epochs: val_acc ≈ 0.54 → 0.70
* Mid training: val_acc ≈ 0.80 → 0.90
* Final epochs: val_acc ≈ **0.9346** (after 30 epochs)

A short sample of inference output printed by the notebook:

```
pred: cat       | true: cat
pred: ship      | true: ship
pred: ship      | true: ship
pred: airplane  | true: airplane
pred: frog      | true: frog
pred: frog      | true: frog
pred: automobile| true: automobile
pred: frog      | true: frog
```

## Checkpointing & logs

* The notebook saves `last.pth` after every epoch and updates `best.pth` when validation accuracy improves. On interrupt (Ctrl+C) it saves `interrupt.pth`.
* The code sets up a `LOG_DIR` path for TensorBoard (`SummaryWriter` is imported).

## How to run

### As a notebook

1. Clone the repo and open the notebook with Jupyter Lab / Notebook.
2. Make sure you have the dependencies.
3. Run cells top to bottom. The notebook contains a self‑contained training loop.

## Reproducibility notes

* The notebook sets `random.seed(SEED)`, `np.random.seed(SEED)`, `torch.manual_seed(SEED)` and `torch.cuda.manual_seed_all(SEED)`.
* `num_workers` is conservatively chosen to keep notebooks responsive (0 on Windows). This affects reproducibility of dataloader ordering.
* Deterministic behavior for CUDA operations requires additional flags (`torch.use_deterministic_algorithms(True)`) and may hurt performance.

## Contact

I'm open to contract and full-time opportunities. Connect with me on LinkedIn: [Nat Andrew](https://www.linkedin.com/in/natandrew).

---
