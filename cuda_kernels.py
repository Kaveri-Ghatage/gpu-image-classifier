"""
Custom GPU Kernel - Fused Image Normalization (written in Triton)
Author: Kaveri Anil Ghatage | github.com/Kaveri-Ghatage

WHY TRITON, so you can explain this in an interview:
Triton is OpenAI's Python-based language/compiler for writing custom GPU
kernels - it compiles directly to PTX (NVIDIA GPU machine code), the same
target raw CUDA C++ compiles to. It's not "fake CUDA" - it's the modern way
serious ML teams write custom kernels (it's literally what powers
torch.compile under the hood). Writing it in Triton instead of raw .cu files
means no separate nvcc build step, and it's genuinely what most production
ML engineers reach for today over hand-written CUDA C++.

WHAT THIS KERNEL DOES:
Normal PyTorch normalization is two separate GPU operations:
  1) subtract the per-channel mean  (one kernel launch, full memory read+write)
  2) divide by the per-channel std  (another kernel launch, full memory read+write)
That means the GPU reads and writes the entire image tensor from memory TWICE.
This kernel fuses both steps into ONE kernel launch - the GPU reads each pixel
once, does (x - mean) / std in registers, and writes it once. Half the memory
traffic for the same math = the speedup.

WHY AUTOTUNING (the @triton.autotune decorator below), a step beyond a
basic kernel: the best BLOCK_SIZE/num_warps for a kernel depends on the
exact GPU it runs on (an RTX 4060 vs a Colab T4 vs an A100 all differ).
Instead of hardcoding one guess, Triton benchmarks several configurations
the first time it sees a new input shape, caches the fastest one, and
reuses it on every later call. This is a real performance-engineering
practice used in production kernels, not just a toy example.
"""
import time
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = torch.cuda.is_available()
    if not TRITON_AVAILABLE:
        print("[INFO] Triton installed but no CUDA GPU detected - kernel needs a GPU to run.")
except ImportError:
    TRITON_AVAILABLE = False
    print("[INFO] Triton not installed. Run: pip install triton (Linux/Colab only)")


if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _fused_normalize_kernel(
        x_ptr, mean_ptr, std_ptr, out_ptr,
        n_elements, HW, C,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)

        # For a contiguous NCHW tensor, the channel of a flat index is
        # (idx // HW) % C  -  e.g. idx = n*(C*HW) + c*HW + h*W + w
        channel = (offsets // HW) % C
        mean = tl.load(mean_ptr + channel, mask=mask)
        std = tl.load(std_ptr + channel, mask=mask)

        y = (x - mean) / (std + 1e-8)
        tl.store(out_ptr + offsets, y, mask=mask)


def fused_normalize_triton(x, mean, std):
    """x: (N,C,H,W) CUDA float tensor. mean/std: (C,) CUDA float tensors."""
    assert x.is_cuda, "Triton kernels only run on CUDA tensors"
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    n_elements = x.numel()
    out = torch.empty_like(x)
    # Note: BLOCK_SIZE is no longer passed manually - @triton.autotune picks
    # the fastest config from the list above based on n_elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_normalize_kernel[grid](x, mean, std, out, n_elements, HW, C)
    return out


class FusedNormalize(nn.Module):
    """
    Drop-in replacement for (x - mean) / std using the custom Triton kernel
    above when on GPU. Falls back to plain PyTorch ops on CPU (correctness
    only - the speedup claim is GPU-only, by definition of "GPU kernel").
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, mean=None, std=None):
        super().__init__()
        mean = mean or self.IMAGENET_MEAN
        std = std or self.IMAGENET_STD
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        if x.is_cuda and TRITON_AVAILABLE:
            return fused_normalize_triton(x, self.mean.to(x.device), self.std.to(x.device))
        m = self.mean.view(1, -1, 1, 1).to(x.device)
        s = self.std.view(1, -1, 1, 1).to(x.device)
        return (x - m) / (s + 1e-8)


def correctness_check(batch=8):
    """Run this first on your GPU - confirms the kernel produces the exact
    same output as the naive PyTorch version before trusting any benchmark."""
    dev = torch.device("cuda")
    x = torch.randn(batch, 3, 224, 224, device=dev)
    mean = torch.tensor([.485, .456, .406], device=dev)
    std = torch.tensor([.229, .224, .225], device=dev)

    triton_out = fused_normalize_triton(x, mean, std)
    naive_out = (x - mean.view(1, -1, 1, 1)) / (std.view(1, -1, 1, 1) + 1e-8)

    match = torch.allclose(triton_out, naive_out, atol=1e-5)
    max_diff = (triton_out - naive_out).abs().max().item()
    print(f"Correctness check: {'PASS' if match else 'FAIL'} (max diff: {max_diff:.2e})")
    return match


def benchmark_one(batch, dev, iters=200):
    x = torch.randn(batch, 3, 224, 224, device=dev)
    mean = torch.tensor([.485, .456, .406], device=dev).view(1, -1, 1, 1)
    std = torch.tensor([.229, .224, .225], device=dev).view(1, -1, 1, 1)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(10):
        _ = (x - mean) / (std + 1e-8)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = (x - mean) / (std + 1e-8)
    sync()
    seq = (time.perf_counter() - t0) / iters * 1000

    norm = FusedNormalize().to(dev)
    for _ in range(10):
        norm(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        norm(x)
    sync()
    fused = (time.perf_counter() - t0) / iters * 1000
    return seq, fused


def benchmark(batches=(8, 16, 32, 64, 128)):
    """Sweeps across batch sizes instead of testing just one - shows whether
    the speedup holds up as batch size grows, which is what a reviewer would
    actually ask about rather than trusting a single number."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'Batch':>6} | {'Sequential (ms)':>16} | {'Fused Triton (ms)':>18} | {'Speedup':>8}")
    print("-" * 58)
    results = []
    for b in batches:
        seq, fused = benchmark_one(b, dev)
        speedup = seq / fused
        results.append((b, seq, fused, speedup))
        print(f"{b:>6} | {seq:>16.3f} | {fused:>18.3f} | {speedup:>7.1f}x")
    return results


if __name__ == "__main__":
    print("Custom GPU Kernel Normalization (Triton, autotuned) - Kaveri Anil Ghatage\n")
    if TRITON_AVAILABLE:
        correctness_check()
        print()
        benchmark()
    else:
        print("Run this on a CUDA GPU (e.g. Google Colab T4, or your RTX 4060) to see real results.")
        print("pip install triton   (Linux only - use WSL2 or Colab on Windows)")
