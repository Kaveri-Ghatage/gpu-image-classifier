"""
GPU-Accelerated Image Classifier - Training
Author: Kaveri Anil Ghatage | github.com/Kaveri-Ghatage

ADVANCED FEATURES (so you can explain each one in an interview):
- Transfer learning on ResNet-50 or EfficientNet-B0 (switchable via --model)
- Data augmentation (RandomResizedCrop, flip, color jitter) - reduces overfitting
  by showing the model slightly different versions of each image every epoch
- FP16 mixed precision via torch.amp - trains faster and uses less GPU memory
  by doing most math in 16-bit instead of 32-bit float
- OneCycleLR schedule - ramps the learning rate up then down over training;
  empirically trains faster and to a better final accuracy than a flat LR
- Per-epoch metrics logged to CSV + loss/accuracy curves saved as a PNG -
  this PNG is your real "Sample Output" proof for your README/portfolio
- Confusion matrix + classification report on validation data - shows
  per-class performance, not just one headline accuracy number
- Saves only the best checkpoint (by validation accuracy), not every epoch
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(name, num_classes):
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Freeze everything except the last residual block + final classifier -
        # this is "partial fine-tuning": faster to train than the full network,
        # and usually enough since early layers already learned general edges/
        # textures from ImageNet pretraining.
        for name_, param in model.named_parameters():
            if not name_.startswith("layer4") and not name_.startswith("fc"):
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for name_, param in model.named_parameters():
            if not name_.startswith("features.8") and not name_.startswith("classifier"):
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def get_dataloaders(data_dir, img_size, batch_size, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="imagenette2-160")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, classes = get_dataloaders(args.data_dir, args.img_size, args.batch_size)
    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(args.model, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    best_val_acc = 0.0
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, scheduler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "epoch_time": epoch_time})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "model_name": args.model, "val_acc": val_acc},
                       "checkpoints/best_model.pth")

    with open("training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs_range = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_range, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs_range, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs_range, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs_range, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=130)
    print("Saved training_curves.png")

    print("\nGenerating classification report on validation set...")
    from sklearn.metrics import classification_report, confusion_matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=classes))
    cm = confusion_matrix(all_labels, all_preds)

    fig2, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig2.colorbar(im)
    fig2.tight_layout()
    fig2.savefig("confusion_matrix.png", dpi=130)
    print("Saved confusion_matrix.png")

    print(f"\nFINAL VALIDATION ACCURACY: {best_val_acc*100:.2f}%")
    print("Best checkpoint saved to checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
