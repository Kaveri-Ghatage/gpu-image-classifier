"""
Fair GPU vs CPU benchmark for a tabular preprocessing pipeline: normalise, filter, join, group-by, sort.
Reports two numbers: compute only (data already on the GPU, like a real pipeline) and end to end
(including copying the table to the GPU and back). Run on a CUDA GPU (e.g. Colab T4):
    python gpu_pipeline_benchmark.py
"""
import sys
import time

import numpy as np
import pandas as pd

try:
    import cudf
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False


def make_df(n, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in "abcd"})
    df["key"] = rng.integers(0, 1000, n)
    return df


def pipeline(df, lookup):
    """Works on pandas and cudf alike."""
    for c in "abcd":
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    df = df[df["a"] > -1.0]
    df = df.merge(lookup, on="key", how="left")
    out = df.groupby("key").agg({"a": ["mean", "std"], "b": "max", "c": "min", "w": "mean"})
    out.columns = ["a_mean", "a_std", "b_max", "c_min", "w_mean"]
    return out.sort_values("a_mean")


def timed(fn, gpu=False):
    if gpu:
        cp.cuda.Device().synchronize()
    t0 = time.time()
    fn()
    if gpu:
        cp.cuda.Device().synchronize()
    return time.time() - t0


def main(sizes):
    lookup = pd.DataFrame({"key": np.arange(1000), "w": np.random.default_rng(1).random(1000)})
    print(f"{'Rows':>11} | {'CPU (s)':>8} | {'GPU compute (s)':>15} | {'speedup':>7} | {'GPU end-to-end (s)':>18} | {'speedup':>7}")
    print("-" * 85)
    if GPU:
        pipeline(cudf.from_pandas(make_df(10_000)), cudf.from_pandas(lookup))  # warm-up
    for n in sizes:
        df = make_df(n)
        cpu = timed(lambda: pipeline(df.copy(), lookup))
        if not GPU:
            print(f"{n:>11,} | {cpu:>8.3f} | {'no GPU':>15} |")
            continue
        gdf, glook = cudf.from_pandas(df), cudf.from_pandas(lookup)
        gpu = timed(lambda: pipeline(gdf.copy(), glook), gpu=True)
        e2e = timed(lambda: pipeline(cudf.from_pandas(df), glook).to_pandas(), gpu=True)
        print(f"{n:>11,} | {cpu:>8.3f} | {gpu:>15.3f} | {cpu / gpu:>6.1f}x | {e2e:>18.3f} | {cpu / e2e:>6.1f}x")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1_000_000, 5_000_000, 10_000_000, 20_000_000])
