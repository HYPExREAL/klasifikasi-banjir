# HydroVision AI: Model Klasifikasi Citra Banjir (MobileNetV3-Large)

Repositori pengembangan model cerdas berbasis **Deep Learning Computer Vision (MobileNetV3-Large)** untuk klasifikasi dan verifikasi otomatis citra banjir. Model ini dirancang khusus untuk mendeteksi genangan air dan luapan banjir dengan akurasi tinggi dan waktu inferensi ultra cepat (~15-25 ms).

---

## 🚀 Fitur Utama

1. **Model AI MobileNetV3-Large Transfer Learning**:
   - Arsitektur ringan, efisien, dan memiliki waktu inferensi sub-detik (hanya ~15-25 ms per gambar).
   - Akurasi pengujian (*Test Accuracy*): **94.32%** pada 1.180 sampel citra uji independen.
   - Sensitivitas deteksi banjir (*Recall*): **95.29%** (meminimalkan *False Negative* agar tidak ada laporan banjir sungguhan yang terlewat).
   - Ekspor model ke format **ONNX Runtime** (`models/flood_mobilenetv3.onnx`) untuk deployment backend yang hemat memori dan cepat.

2. **Dashboard Admin Pintar (Smart Sorting & Filtering)**:
   - **Kartu Metrik Real-Time**: Total Laporan Masuk, Prioritas Banjir (≥85%), Perlu Tinjauan Manual (50–85%), Bukan Banjir / Spam, dan Titik Aktif di Peta.
   - **Filter Cepat**: Menyortir laporan berdasarkan keputusan dan keyakinan AI.
   - **One-Click Action**: Admin dapat langsung menyetujui (*Approve*) laporan untuk diterbitkan ke peta publik atau menolak (*Reject*) laporan palsu/spam.

3. **Peta Titik Banjir Interaktif (GIS Leaflet.js)**:
   - Visualisasi titik-titik banjir aktif yang telah terverifikasi oleh AI dan disetujui admin.
   - Pop-up interaktif menampilkan foto lokasi, alamat, keterangan warga, dan persentase keyakinan AI.

4. **Simulasi Pelaporan Masyarakat**:
   - Formulir unggah foto dengan **Live AI Inference Scan** seketika (*instant preview*) sebelum dikirim ke database.

---

## 📁 Struktur Direktori

```text
klasifikasi-banjir/
├── dataset/                    # Dataset citra (6.180 gambar)
│   ├── train/                  # 4.000 gambar (2.000 banjir, 2.000 bukan_banjir)
│   ├── val/                    # 1.000 gambar (500 banjir, 500 bukan_banjir)
│   └── test/                   # 1.180 gambar (552 banjir, 628 bukan_banjir)
├── models/
│   ├── best_mobilenet_v3.pth   # Checkpoint model PyTorch terbaik (Val Acc: 97.10%)
│   ├── flood_mobilenetv3.onnx  # Model ONNX untuk inferensi produksi
│   ├── class_indices.json      # Mapping index label {'banjir': 0, 'bukan_banjir': 1}
│   ├── confusion_matrix.png    # Visualisasi matriks konfusi
│   └── test_evaluation_results.json # Hasil evaluasi lengkap pada data test
├── scripts/
│   ├── split_dataset.py        # Skrip pembagian validation set
│   ├── verify_images.py        # Pemeriksaan integritas seluruh citra
│   ├── train.py                # Pipeline pelatihan MobileNetV3-Large
│   ├── evaluate.py             # Evaluasi test set & kalkulasi metrik
│   ├── export_onnx.py          # Konversi model PyTorch ke format ONNX
│   ├── predict.py              # Inferensi cepat pada citra tunggal
│   ├── collect_hard_negatives.py # Deteksi sampel hard negative
│   ├── finetune_hard_negatives.py # Fine-tuning model dengan sampel sulit
│   ├── plot_confusion_matrix.py # Pembuatan visualisasi matriks konfusi
│   └── generate_docx_document.py # Generator dokumen laporan metodologi .docx
├── DOKUMEN_METODOLOGI_MODEL_AI_TUGAS_AKHIR.docx # Dokumen metodologi model AI resmi
└── README.md                   # Dokumentasi proyek
```

---

## ⚙️ Cara Menjalankan Skrip Pengembangan AI

Pastikan virtual environment aktif, kemudian jalankan perintah sesuai kebutuhan:

- **Prediksi Citra Tunggal**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/predict.py "dataset/test/banjir/flooded_0003.jpg"
  ```

- **Evaluasi Data Uji (*Test Set*)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/evaluate.py
  ```

- **Ekspor Model ke Format ONNX**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/export_onnx.py
  ```

- **Pelatihan (*Training*) Model**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/train.py
  ```

- **Fine-Tuning Sampel Sulit (*Hard Negatives*)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/finetune_hard_negatives.py
  ```

- **Generate Dokumen Metodologi Tugas Akhir (.docx)**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/generate_docx_document.py
  ```

---

## 📊 Hasil Evaluasi Model

| Metrik | Kelas Banjir | Kelas Bukan Banjir | Keseluruhan |
| :--- | :---: | :---: | :---: |
| **Precision** | **92.77%** | **95.76%** | - |
| **Recall** | **95.29%** | **93.47%** | - |
| **F1-Score** | **94.01%** | **94.60%** | - |
| **Akurasi Total** | - | - | **94.32%** (1.113 / 1.180 benar) |

### Confusion Matrix Test Set:
```text
Aktual \ Prediksi     Banjir       Bukan Banjir
------------------------------------------------
Banjir                 526             26
Bukan Banjir            41            587
```
