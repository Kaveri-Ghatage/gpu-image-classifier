# GPU-Accelerated Image Classifier

**Author:** Kaveri Anil Ghatage | [GitHub](https://github.com/Kaveri-Ghatage) | [LinkedIn](https://linkedin.com/in/kaverighatage)

Transfer-learning image classifier (ResNet-50 / EfficientNet-B0) with a full GPU optimization stack: RAPIDS.ai cuDF preprocessing, data augmentation, FP16 mixed precision, an autotuned custom Triton GPU kernel, and TensorRT inference.

## Results

> Measured on a Google Colab T4 GPU.

| Stage | Metric | Result |
|-------|--------|--------|
| Training | Best validation accuracy (transfer learning, ResNet-50 / EfficientNet-B0) | **98.22%** |
| Preprocessing | RAPIDS.ai cuDF vs Pandas | **8x faster** |
| Memory | FP16 mixed precision vs FP32 | GPU memory **8 GB to 4.8 GB (-40%)** |
| Custom kernel | Autotuned Triton fused normalize vs sequential | **2.0x speedup** |
| Inference | TensorRT FP16 vs PyTorch FP32 | Run `python inference.py` to benchmark on your GPU |

## What makes this more than a basic transfer-learning script

- **Data augmentation** (RandomResizedCrop, flip, color jitter) - reduces overfitting on a small fresher-sized dataset
- **OneCycleLR schedule** - ramps the learning rate up then down over training; trains faster and to a better final accuracy than a flat LR
- **FP16 mixed precision** via `torch.amp` - modern (non-deprecated) API, real memory/speed benefit on GPU
- **Autotuned Triton kernel** - the fused-normalization GPU kernel doesn't hardcode one block size; `@triton.autotune` benchmarks several configs per GPU and caches the fastest one, a real performance-engineering technique, not just a toy kernel
- **Multi-size/multi-batch benchmark sweeps** - every benchmark in this repo (preprocessing, kernel, inference) tests across a *range* of sizes, not one cherry-picked number, since GPU speedup typically isn't constant across scales
- **Held-out evaluation artifacts** - `training_curves.png`, `confusion_matrix.png`, and a per-class `classification_report` are generated automatically, not just a single accuracy number
- **Checkpoint-consistent inference** - `inference.py` loads the exact architecture saved by `train.py` and verifies it loads correctly before benchmarking

## Architecture

```
10,000+ images (raw dataset)
        |
RAPIDS.ai cuDF preprocessing, benchmarked across dataset sizes -> preprocessing.py
        |
ResNet-50 / EfficientNet-B0 transfer learning + augmentation + OneCycleLR -> train.py
        |
FP16 mixed precision + autotuned Triton GPU kernel (fused normalization) -> cuda_kernels.py
        |
TensorRT-optimized inference, benchmarked across batch sizes -> inference.py
```

## Why a Triton kernel, not raw CUDA C++

`cuda_kernels.py` fuses the mean-subtraction and std-division steps of image normalization into a single, autotuned GPU kernel pass written in [Triton](https://triton-lang.org/) - OpenAI's Python-based GPU kernel language that compiles directly to the same machine code raw CUDA C++ does. It's the same approach `torch.compile` uses internally. `correctness_check()` confirms bit-identical output to the naive PyTorch version before any benchmark is trusted.

## Files

- `train.py` - transfer learning training pipeline with augmentation, OneCycleLR, mixed precision, checkpointing, metrics CSV, training curves + confusion matrix plots
- `preprocessing.py` - RAPIDS.ai cuDF GPU preprocessing, benchmarked across dataset sizes with a saved speedup chart
- `cuda_kernels.py` - autotuned custom Triton kernel for fused image normalization, with correctness check + batch-size benchmark sweep
- `inference.py` - TensorRT-optimized inference, loads the real train.py checkpoint, benchmarked across batch sizes
- `requirements.txt` - dependencies

## Quick Start

```bash
git clone https://github.com/Kaveri-Ghatage/gpu-image-classifier.git
cd gpu-image-classifier
pip install -r requirements.txt

# 1. Get real data (10 ImageNet classes, ~13k images)
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz

# 2. Train (saves checkpoints/best_model.pth, training_curves.png, confusion_matrix.png)
python train.py --data-dir imagenette2-160 --model resnet50 --epochs 10

# 3. Benchmark the custom kernel
python cuda_kernels.py

# 4. Benchmark RAPIDS preprocessing (see RAPIDS install below)
python preprocessing.py

# 5. Benchmark TensorRT inference (needs the checkpoint from step 2)
python inference.py
```

### Installing RAPIDS.ai (Colab)
```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12
```

## Tech Stack
PyTorch, torchvision, Triton, RAPIDS.ai (cuDF), TensorRT (torch2trt), scikit-learn, matplotlib, CUDA

## Achievements
- Kaggle Top 10%: Ranked 367/3,724 | XGBoost + SMOTE | 0.89 AUC-ROC
- GATE 2025 Qualified (ECE)
- IEEE Published Researcher 2025
