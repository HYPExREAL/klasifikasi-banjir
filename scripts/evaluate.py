import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def evaluate_model(
    model_path="models/best_mobilenet_v3.pth",
    test_dir="dataset/test",
    output_dir="models",
    batch_size=32,
    device_name=None
):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Evaluasi menggunakan: {device}")

    out_path = Path(output_dir)
    test_path = Path(test_dir)

    # 1. Transformasi Data Test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = test_dataset.classes
    print(f"[DATASET TEST] Kelas: {classes}")
    print(f"               Total Sampel: {len(test_dataset)}")

    # 2. Muat Model
    print(f"\n[MODEL] Memuat checkpoint dari {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 3. Prediksi pada Test Set
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    # 4. Hitung Metrik Evaluasi
    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / total

    # Confusion matrix manual (2x2)
    # 0 = banjir, 1 = bukan_banjir (atau sesuai urutan alfabetis classes)
    cm = [[0 for _ in classes] for _ in classes]
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1

    print("\n" + "=" * 60)
    print("HASIL EVALUASI TEST SET (MOBILENETV3-LARGE)")
    print("=" * 60)
    print(f"Akurasi Keseluruhan: {accuracy * 100:.2f}% ({correct}/{total} benar)")
    print("\nConfusion Matrix:")
    header = f"{'Aktual \\ Prediksi':<20}" + "".join([f"{cls:<15}" for cls in classes])
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(classes):
        row = f"{cls:<20}" + "".join([f"{cm[i][j]:<15}" for j in range(len(classes))])
        print(row)

    # Metrik per kelas
    metrics_per_class = {}
    print("\nMetrik Performa per Kelas:")
    print(f"{'Kelas':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)

    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(classes)) if r != i)
        fn = sum(cm[i][c] for c in range(len(classes)) if c != i)
        support = sum(cm[i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics_per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
        print(f"{cls:<15} {prec*100:>8.2f}%    {rec*100:>8.2f}%    {f1*100:>8.2f}%    {support:<10}")

    print("=" * 60)

    # Simpan hasil evaluasi
    eval_results = {
        "accuracy": round(accuracy, 4),
        "total_test_samples": total,
        "classes": classes,
        "confusion_matrix": cm,
        "per_class": metrics_per_class
    }
    with open(out_path / "test_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"[INFO] Hasil evaluasi disimpan ke: {out_path / 'test_evaluation_results.json'}")

if __name__ == "__main__":
    evaluate_model()
