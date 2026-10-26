"""
TensorRT-Optimized Inference
Author: Kaveri Anil Ghatage | github.com/Kaveri-Ghatage

Loads the checkpoint saved by train.py, benchmarks PyTorch FP32 vs a
TensorRT FP16 engine across multiple batch sizes (single-batch numbers
can be misleading - the question that actually matters for a real
deployment is "how does this hold up at the batch size I'll serve at").
"""
import os, time
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    import torch2trt
    TRT = True
    print("[INFO] TensorRT (torch2trt) available")
except ImportError:
    TRT = False
    print("[INFO] TensorRT not installed - PyTorch-only benchmark. "
          "On Colab: pip install torch2trt (or build from github.com/NVIDIA-AI-IOT/torch2trt)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INF_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


def build_model_for_inference(model_name, num_classes):
    """Mirrors build_model() in train.py exactly - the architecture must
    match what was actually trained, or load_state_dict will fail."""
    if model_name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model_name = ckpt["model_name"]
    model = build_model_for_inference(model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"[INFO] {model_name} loaded | val_acc={ckpt.get('val_acc', 0)*100:.2f}%")
    return model, classes


def to_tensorrt(model, batch=1):
    if not TRT:
        return model
    dummy = torch.randn(batch, 3, 224, 224, device=device)
    try:
        trt_m = torch2trt.torch2trt(model, [dummy], fp16_mode=True, max_workspace_size=1 << 30)
        print("[INFO] TensorRT engine compiled")
        return trt_m
    except Exception as e:
        print(f"[WARN] TRT compile failed: {e}")
        return model


def predict(model, path, classes):
    img = Image.open(path).convert("RGB")
    t = INF_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        probs = torch.softmax(model(t), 1)
        conf, idx = probs.max(1)
    return {"class": classes[idx.item()], "confidence": conf.item() * 100}


def benchmark_one(model, batch, iters=100):
    x = torch.randn(batch, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / iters * 1000
    throughput = batch / (lat / 1000)
    return lat, throughput


def benchmark_sweep(model, batches=(1, 8, 32, 64)):
    print(f"{'Batch':>6} | {'Latency (ms)':>13} | {'Throughput (img/s)':>19}")
    print("-" * 46)
    results = []
    for b in batches:
        lat, tp = benchmark_one(model, b)
        print(f"{b:>6} | {lat:>13.2f} | {tp:>19.0f}")
        results.append((b, lat, tp))
    return results


if __name__ == "__main__":
    print("TensorRT Inference Benchmark - Kaveri Anil Ghatage\n")
    ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        model, classes = load_model(ckpt_path)

        print("\n--- PyTorch FP32 ---")
        pt_results = benchmark_sweep(model)

        if TRT and device.type == "cuda":
            trt_model = to_tensorrt(model)
            print("\n--- TensorRT FP16 ---")
            trt_results = benchmark_sweep(trt_model)

            print(f"\n{'Batch':>6} | {'Speedup':>8}")
            print("-" * 18)
            for (b, pt_lat, _), (_, trt_lat, _) in zip(pt_results, trt_results):
                print(f"{b:>6} | {pt_lat/trt_lat:>7.1f}x")
        elif not TRT:
            print("\nInstall torch2trt to benchmark TensorRT speedup.")
        else:
            print("\nTensorRT requires a CUDA GPU.")
    else:
        print(f"No checkpoint found at {ckpt_path}. Run train.py first.")
