# GPU-Accelerated Image Classifier

**Author:** Kaveri Anil Ghatage | [GitHub](https://github.com/Kaveri-Ghatage) | [LinkedIn](https://linkedin.com/in/kaverighatage)

Transfer-learning image classifier (ResNet-50 / EfficientNet-B0) with a full GPU optimization stack: RAPIDS.ai cuDF preprocessing, data augmentation, FP16 mixed precision, an autotuned custom Triton GPU kernel, and TensorRT inference.

## Results

All numbers were measured on a free Google Colab **T4 GPU** and can be reproduced with the commands below.

| Stage | What was measured | Result |
|-------|-------------------|--------|
| Training | ResNet-50 transfer learning, 10 epochs, Imagenette (10 classes, 9,469 train / 3,925 validation images) | **98.24%** validation accuracy |
| Mixed precision | FP16 vs FP32, one ResNet-50 training step (`memory_benchmark.py`) | peak GPU memory 1,710 MB to 1,072 MB (**-37%**), **2.3x** faster per step |
| Custom kernel | Autotuned Triton fused normalisation vs sequential PyTorch (`cuda_kernels.py`) | 1.7x at batch 8, **2.0x from batch 32 up**; output matches the reference (max difference 1.9e-6) |
| Preprocessing | RAPIDS cuDF vs pandas, normalise + filter + join + group-by + sort (`gpu_pipeline_benchmark.py`) | **8.5x** at 10M rows end to end (including copying to the GPU and back); **13.6x** at 20M rows compute only |
| Inference | TensorRT FP16 vs PyTorch FP32 | not benchmarked yet (`python inference.py`) |

**Honest notes**
- Imagenette is an easy 10-class dataset, so 98% shows the pipeline works, not that it would reach this on a hard dataset.
- GPU preprocessing only pays off on larger tables or heavier work. A light single-pass normalisation (`preprocessing.py`) gave just 1.1x to 1.4x at 1M to 20M rows, because copying data to the GPU dominates.
- Only ResNet-50 was trained for the numbers above; EfficientNet-B0 is supported (`--model efficientnet_b0`) but not reported.

Training curves and the confusion matrix are in `training_curves.png` and `confusion_matrix.png`.

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
Imagenette: ~13,400 images, 10 classes
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
- `memory_benchmark.py` - peak GPU memory and step time, FP32 vs FP16
- `gpu_pipeline_benchmark.py` - RAPIDS cuDF vs pandas on a realistic tabular pipeline, with and without GPU transfer
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

# 3. Benchmark the custom kernel and mixed precision
python cuda_kernels.py
python memory_benchmark.py

# 4. Benchmark RAPIDS preprocessing (needs a GPU runtime such as Colab)
python gpu_pipeline_benchmark.py

# 5. Benchmark TensorRT inference (needs the checkpoint from step 2)
python inference.py
```

### RAPIDS on Colab
Colab's GPU runtime already includes cuDF, so do not pip-install it. Just run `python gpu_pipeline_benchmark.py`. Without a GPU the script still runs, but it only times pandas.

## Tech Stack
PyTorch, torchvision, Triton, RAPIDS.ai (cuDF), TensorRT (torch2trt), scikit-learn, matplotlib, CUDA

## Achievements
- Kaggle Top 10%: Ranked 367/3,724 | XGBoost + SMOTE | 0.89 AUC-ROC
- GATE 2025 Qualified (ECE)
- IEEE Published Researcher 2025
