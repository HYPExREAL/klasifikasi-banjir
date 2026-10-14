"""
Skrip untuk menampilkan dan menghasilkan visualisasi Confusion Matrix & Classification Report
sesuai format evaluasi model klasifikasi banjir MobileNetV3-Large.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def main():
    # Load hasil evaluasi
    json_path = Path("models/test_evaluation_results.json")
    if not json_path.exists():
        print(f"Error: {json_path} tidak ditemukan.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data["confusion_matrix"]
    classes = [c.replace("_", " ").title() for c in data["classes"]]
    # classes: ['Banjir', 'Bukan Banjir']

    tp = cm[0][0]  # Banjir diprediksi Banjir: 526
    fn = cm[0][1]  # Banjir diprediksi Bukan Banjir: 26
    fp = cm[1][0]  # Bukan Banjir diprediksi Banjir: 41
    tn = cm[1][1]  # Bukan Banjir diprediksi Bukan Banjir: 587

    total_samples = data["total_test_samples"]
    total_correct = tp + tn
    accuracy = data["accuracy"]

    # Buat array y_true dan y_pred untuk sklearn classification_report
    y_true = [0] * (tp + fn) + [1] * (fp + tn)
    y_pred = [0] * tp + [1] * fn + [0] * fp + [1] * tn

    # 1. CETAK KE TERMINAL SESUAI FORMAT SCREENSHOT
    print(f"Total foto berhasil diuji: {total_samples}")
    print("=== CONFUSION MATRIX ===")
    print(f"{'':22}{'Prediksi: Banjir':>18}{'Prediksi: Bukan Banjir':>24}")
    print(f"{'Aktual: Banjir':<22}{tp:>18}{fn:>24}")
    print(f"{'Aktual: Bukan Banjir':<22}{fp:>18}{tn:>24}")
    print()
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print()
    print("=== KOMPOSISI SAMPEL PER KELAS ===")
    print(f"{classes[0]:<16}{tp + fn}")
    print(f"{classes[1]:<16}{fp + tn}")
    print("Name: count, dtype: int64")
    print(f"Jumlah sampel: {total_samples}")
    print(f"Jumlah benar : {total_correct}")
    print(f"Akurasi kasar: {accuracy:.4f}")

    # 2. GENERATE GRAFIK HEATMAP VISUAL
    cm_array = np.array(cm)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - HydroVision (MobileNetV3-Large)", fontsize=13, pad=16, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [f"Pred: {c}" for c in classes], fontsize=11)
    plt.yticks(tick_marks, [f"Aktual: {c}" for c in classes], fontsize=11, rotation=0)

    # Tambahkan teks angka & persentase di dalam cell
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        row_total = np.sum(cm_array[i, :])
        for j in range(cm_array.shape[1]):
            val = cm_array[i, j]
            pct = (val / row_total) * 100
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}\n({pct:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=color, fontsize=12, fontweight='bold')

    plt.ylabel("Label Aktual", fontsize=12, labelpad=10, fontweight='bold')
    plt.xlabel("Label Prediksi", fontsize=12, labelpad=10, fontweight='bold')
    plt.tight_layout()

    out_image_path = Path("models/confusion_matrix.png")
    plt.savefig(out_image_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[INFO] Gambar grafik Confusion Matrix berhasil disimpan di: {out_image_path}")

if __name__ == "__main__":
    main()
