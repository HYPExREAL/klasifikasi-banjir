import os
import sys
import time
import json
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.export_onnx import export_to_onnx
from scripts.evaluate import evaluate_model

def finetune_model(
    data_dir="dataset",
    output_dir="models",
    pretrained_checkpoint="models/best_mobilenet_v3.pth",
    epochs=4,
    batch_size=32,
    lr=2e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("FINETUNING MODEL MOBILENETV3-LARGE DENGAN HARD NEGATIVES")
    print("=" * 65)
    print(f"[DEVICE] Perangkat: {device}")
    
    out_path = Path(output_dir)
    data_path = Path(data_dir)

    # 1. Transformasi Data dengan Augmentasi yang Diperkuat
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(data_path / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(data_path / "val", transform=val_transforms)

    print(f"[DATASET] Kelas        : {train_dataset.classes}")
    print(f"          Mapping      : {train_dataset.class_to_idx}")
    print(f"          Total Train  : {len(train_dataset)} citra")
    print(f"          Total Val    : {len(val_dataset)} citra")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Muat Checkpoint Model Sebelumnya
    print(f"\n[MODEL] Memuat bobot awal dari: {pretrained_checkpoint}")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(train_dataset.classes))

    if Path(pretrained_checkpoint).exists():
        ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print("        Bobot checkpoint berhasil dimuat!")
    else:
        print("        [WARN] Checkpoint tidak ditemukan, menggunakan ImageNet pre-trained.")
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(in_features, len(train_dataset.classes))

    model = model.to(device)

    # 3. Loss Berbobot (Memberi sedikit penalti lebih tinggi pada false positive bukan banjir)
    # class 0: banjir (1.0), class 1: bukan_banjir (1.2)
    weights = torch.tensor([1.0, 1.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_file = out_path / "best_mobilenet_v3.pth"
    total_batches = len(train_loader)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

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

            if batch_idx % 30 == 0 or batch_idx == total_batches:
                curr_loss = running_loss / total_train
                curr_acc = (running_corrects / total_train) * 100
                print(f"  [Epoch {epoch}/{epochs}] Batch {batch_idx:03d}/{total_batches:03d} | Loss: {curr_loss:.4f} | Acc: {curr_acc:.2f}%", flush=True)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # Validation
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
        scheduler.step()

        duration = time.time() - epoch_start
        print(f"--> [Epoch {epoch}/{epochs}] Selesai ({duration:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.2f}%", flush=True)

        # Selalu simpan atau simpan yang terbaik
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "class_to_idx": train_dataset.class_to_idx
            }, best_model_file)
            print(f"  [SAVE] Checkpoint baru disimpan -> {best_model_file} (Val Acc: {best_val_acc*100:.2f}%)", flush=True)

    print("\n" + "=" * 65)
    print(f"PROSES FINE-TUNING SELESAI ({time.time() - start_time:.1f} detik)")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 65)

    # 4. Otomatis Ekspor ke ONNX
    print("\n[EXPORT] Mengekspor model hasil fine-tuning ke ONNX...")
    export_to_onnx(
        model_path=str(best_model_file),
        output_onnx=str(out_path / "flood_mobilenetv3.onnx"),
        classes_json=str(out_path / "class_indices.json")
    )

    # 5. Otomatis Evaluasi pada Data Test
    print("\n[EVALUASI] Menjalankan evaluasi pada Data Uji (Test Set)...")
    evaluate_model(
        model_path=str(best_model_file),
        test_dir=str(data_path / "test"),
        output_dir=str(out_path)
    )

if __name__ == "__main__":
    finetune_model()
