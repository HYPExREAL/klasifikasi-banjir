import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_mobilenet_v3(
    data_dir="dataset",
    output_dir="models",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    device_name=None
):
    # Setup Device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Menggunakan perangkat: {device}")
    if device.type == "cuda":
        print(f"         GPU: {torch.cuda.get_device_name(0)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Image Transformations
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # 2. Data Loaders
    data_path = Path(data_dir)
    train_dataset = datasets.ImageFolder(data_path / "train", transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(data_path / "val", transform=data_transforms["val"])

    print(f"[DATASET] Kelas yang ditemukan: {train_dataset.classes}")
    print(f"          Index kelas          : {train_dataset.class_to_idx}")
    print(f"          Total Sampel Train   : {len(train_dataset)}")
    print(f"          Total Sampel Val     : {len(val_dataset)}")

    # Simpan mapping kelas ke JSON
    with open(out_path / "class_indices.json", "w") as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)

    # Windows DataLoader: num_workers=0 atau 2
    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model Architecture (MobileNetV3-Large Transfer Learning)
    print("\n[MODEL] Mengunduh bobot pre-trained MobileNetV3-Large...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    # Ganti classifier terakhir: MobileNetV3 classifier[-1] adalah Linear(in_features=1280, out_features=1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(train_dataset.classes))

    model = model.to(device)

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # 5. Training Loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n" + "=" * 65)
    print("MEMULAI TRAINING MODEL MOBILENETV3-LARGE")
    print("=" * 65)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

            if batch_idx % 25 == 0 or batch_idx == total_batches:
                current_acc = (running_corrects / total_train) * 100
                current_loss = running_loss / total_train
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch {batch_idx:03d}/{total_batches:03d} | Train Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%", flush=True)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_corrects / total_val

        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        epoch_duration = time.time() - epoch_start
        print(f"--> Selesai Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc*100:.2f}%", flush=True)

        # Checkpoint Best Model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_file = out_path / "best_mobilenet_v3.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "class_to_idx": train_dataset.class_to_idx
            }, best_model_file)
            print(f"  [SAVE] Checkpoint tersimpan: {best_model_file} (Akurasi Val: {best_val_acc*100:.2f}%)", flush=True)

    total_duration = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"TRAINING SELESAI dalam {total_duration/60:.2f} menit!")
    print(f"Akurasi Validasi Tertinggi: {best_val_acc*100:.2f}%")
    print("=" * 65)

    # Simpan riwayat training
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    train_mobilenet_v3()
