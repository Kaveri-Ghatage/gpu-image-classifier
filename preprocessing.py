"""
GPU-Accelerated Preprocessing - RAPIDS.ai cuDF
Author: Kaveri Anil Ghatage | github.com/Kaveri-Ghatage

Benchmarks RAPIDS.ai cuDF (GPU dataframes) against Pandas (CPU) across
several dataset sizes - a single-size benchmark can be misleading since
GPU speedup typically grows with data size (more work to parallelize
relative to the fixed overhead of moving data to the GPU).
Parallel image loading via Python multiprocessing (8 workers).
"""
import time, multiprocessing as mp
import numpy as np, pandas as pd
from PIL import Image

try:
    import cudf
    RAPIDS = True
    print("[INFO] RAPIDS.ai cuDF available")
except ImportError:
    RAPIDS = False
    print("[INFO] RAPIDS.ai not found - using CPU (Pandas) only. Install for GPU speedup.")

def preprocess_gpu(df):
    if not RAPIDS:
        return preprocess_cpu(df)
    gdf = cudf.from_pandas(df)
    for col in gdf.select_dtypes(include=["float32","float64","int32","int64"]).columns:
        mu, sigma = gdf[col].mean(), gdf[col].std()
        if sigma > 0:
            gdf[col] = (gdf[col] - mu) / sigma
    if "label" in gdf.columns:
        gdf["label_enc"] = gdf["label"].astype("category").cat.codes
    return gdf.fillna(0).to_pandas()

def preprocess_cpu(df):
    out = df.copy()
    for col in out.select_dtypes(include=["float32","float64","int32","int64"]).columns:
        mu, sigma = out[col].mean(), out[col].std()
        if sigma > 0:
            out[col] = (out[col] - mu) / sigma
    if "label" in out.columns:
        out["label_enc"] = pd.Categorical(out["label"]).codes
    return out.fillna(0)

def _load(args):
    path, size = args
    try:
        img = Image.open(path).convert("RGB").resize((size,size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - [.485,.456,.406]) / [.229,.224,.225]
        return arr, True
    except Exception:
        return None, False

def parallel_load(paths, size=224, workers=8):
    """Parallel image preprocessing with 8 workers (resume: multiprocessing 8 workers)."""
    print(f"[INFO] Loading {len(paths)} images | workers={workers}")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        res = pool.map(_load, [(p,size) for p in paths])
    imgs = [r[0] for r in res if r[1]]
    print(f"[INFO] Done: {len(imgs)}/{len(paths)} in {time.time()-t0:.2f}s")
    return np.array(imgs, dtype=np.float32)

def adaptive_batch(base=64):
    """Adaptive batch sizing based on available GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            factor = 4 if free_gb>8 else 2 if free_gb>4 else 1 if free_gb>2 else 0.5
            bs = int(base * factor)
            print(f"[INFO] Adaptive batch size: {bs} (free GPU: {free_gb:.1f}GB)")
            return bs
    except Exception:
        pass
    return base

def _make_df(n, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in ["a", "b", "c", "d"]})
    df["label"] = rng.integers(0, 10, n)
    return df

def benchmark(sizes=(10_000, 50_000, 100_000, 500_000)):
    print(f"{'Rows':>10} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'Speedup':>8}")
    print("-" * 48)
    results = []
    for n in sizes:
        df = _make_df(n)
        t0 = time.time(); preprocess_cpu(df.copy()); cpu = time.time() - t0
        if RAPIDS:
            t0 = time.time(); preprocess_gpu(df.copy()); gpu = time.time() - t0
            speedup = cpu / gpu
            print(f"{n:>10,} | {cpu:>10.3f} | {gpu:>10.3f} | {speedup:>7.1f}x")
            results.append((n, cpu, gpu, speedup))
        else:
            print(f"{n:>10,} | {cpu:>10.3f} | {'N/A':>10} | Install RAPIDS.ai")
            results.append((n, cpu, None, None))

    if RAPIDS and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ns = [r[0] for r in results]
            speedups = [r[3] for r in results]
            plt.figure(figsize=(6, 4))
            plt.plot(ns, speedups, marker="o")
            plt.xlabel("Dataset size (rows)"); plt.ylabel("Speedup (x)")
            plt.title("RAPIDS.ai cuDF Speedup vs Pandas")
            plt.xscale("log")
            plt.tight_layout()
            plt.savefig("preprocessing_speedup.png", dpi=130)
            print("Saved preprocessing_speedup.png")
        except ImportError:
            pass
    return results

if __name__ == "__main__":
    print("RAPIDS.ai GPU Preprocessing - Kaveri Anil Ghatage\n")
    benchmark()
