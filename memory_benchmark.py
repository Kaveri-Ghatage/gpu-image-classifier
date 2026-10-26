"""Measures peak GPU memory and speed of one ResNet-50 training step: FP32 vs FP16 mixed precision."""
import time
import torch
import torchvision


def run(use_amp, steps=20, batch=32, size=160):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = torchvision.models.resnet50(num_classes=10).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    x = torch.randn(batch, 3, size, size, device="cuda")
    y = torch.randint(0, 10, (batch,), device="cuda")
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20, (time.time() - t0) / steps * 1000


if __name__ == "__main__":
    mem32, ms32 = run(False)
    mem16, ms16 = run(True)
    print(f"FP32: peak memory {mem32:.0f} MB, {ms32:.0f} ms/step")
    print(f"FP16: peak memory {mem16:.0f} MB, {ms16:.0f} ms/step")
    print(f"Memory saved: {(1 - mem16 / mem32) * 100:.0f}%   Speedup: {ms32 / ms16:.2f}x")
