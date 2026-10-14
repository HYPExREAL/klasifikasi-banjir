"""
Skrip untuk membuat dokumen Microsoft Word (.docx) resmi dan terformat rapi
untuk dokumentasi teknis Model AI Tugas Akhir.
"""

from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_document():
    doc = Document()

    # Set page margins (Standard 1 inch / 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Style default
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # JUDUL DOKUMEN
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("DOKUMENTASI TEKNIS MODEL AI KLASIFIKASI BANJIR\nMOBILENETV3-LARGE")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x11, 0x33, 0x66)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(20)
    run_sub = sub_p.add_run("Untuk Penyusunan Dokumen Tugas Akhir / Skripsi (BAB III Metodologi & BAB IV Hasil Pembahasan)\nSistem Pelaporan dan Visualisasi Titik Banjir Berbasis WebGIS MERN Stack")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # 1. ARSITEKTUR MODEL
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    r1 = h1.add_run("1. Arsitektur dan Landasan Pemilihan Model (BAB III Metodologi)")
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x11, 0x33, 0x66)

    # 1.A
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(6)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("A. Alasan Pemilihan Arsitektur MobileNetV3-Large")
    r2.font.size = Pt(12)
    r2.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Penelitian ini menerapkan arsitektur MobileNetV3-Large berbasis Deep Learning Convolutional Neural Network (CNN) "
        "dengan pendekatan Transfer Learning. Pemilihan arsitektur ini didasari oleh kebutuhan sistem WebGIS pelaporan "
        "yang memerlukan inferensi real-time, akurasi tinggi, serta efisiensi penggunaan sumber daya server. "
        "Faktor-faktor keunggulan utama meliputi:"
    )

    bullets = [
        ("Efisiensi Komputasi Tinggi (NAS & NetAdapt): ", 
         "MobileNetV3 dirancang melalui kombinasi Neural Architecture Search (NAS) untuk mengoptimalkan struktur blok global dan algoritma NetAdapt untuk menyempurnakan jumlah filter per layer secara lokal."),
        ("Inverted Residuals & Squeeze-and-Excitation (SE): ", 
         "Memanfaatkan blok residual terbalik dengan Depthwise Separable Convolutions yang memangkas beban perkalian parameter komputasi secara signifikan. Dilengkapi modul atensi Squeeze-and-Excitation (SE) yang meningkatkan sensitivitas representasi spasial pada fitur kritis genangan air."),
        ("Fungsi Aktivasi Hard-Swish (h-swish): ", 
         "Menggantikan fungsi sigmoid/swish standar dengan rumus h-swish = x * ReLU6(x + 3) / 6 untuk mempercepat perhitungan perkalian floating-point tanpa mengorbankan diferensiasi gradien."),
        ("Ukuran Model Ramping & Inferensi Sub-Detik: ", 
         "Model hasil ekspor ONNX hanya berukuran ~16.8 MB dengan kecepatan inferensi rata-rata 15-25 ms per citra pada CPU biasa.")
    ]
    for b_title, b_desc in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        bp.paragraph_format.space_after = Pt(3)
        rt = bp.add_run(b_title)
        rt.bold = True
        bp.add_run(b_desc)

    # 1.B
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("B. Pendekatan Transfer Learning & Modifikasi Classifier Head")
    r2.font.size = Pt(12)
    r2.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Model memanfaatkan bobot awal (pre-trained weights) dari dataset ImageNet-1K. "
        "Lapisan ekstraktor fitur (backbone konvolusi) dipertahankan, sedangkan lapisan pengklasifikasi akhir "
        "(classifier head) disesuaikan untuk tugas klasifikasi biner citra banjir:"
    )

    classifier_details = [
        ("Dimensi Input Citra: ", "Tensor (1, 3, 224, 224) dengan normalisasi standar ImageNet (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])."),
        ("Lapisan Regularisasi: ", "Dropout(p=0.2) untuk mencegah ketergantungan fitur berlebih dan mengatasi overfitting."),
        ("Lapisan Linear Akhir: ", "Linear(in_features=1280, out_features=2) yang memetakan fitur ke dalam dua kelas biner [Banjir, Bukan Banjir].")
    ]
    for c_title, c_desc in classifier_details:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        bp.paragraph_format.space_after = Pt(3)
        rt = bp.add_run(c_title)
        rt.bold = True
        bp.add_run(c_desc)

    # 2. PENGOLAHAN DATA & PELATIHAN
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    r1 = h1.add_run("2. Pengolahan Data dan Pelatihan Model (BAB III Metodologi)")
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x11, 0x33, 0x66)

    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(6)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("A. Pembagian Dataset (Dataset Splitting)")
    r2.font.size = Pt(12)
    r2.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Total dataset terdiri dari 6.180 citra yang dibagi ke dalam 3 subset independen secara terstratifikasi (stratified split):"
    )

    # Tabel Dataset
    table_ds = doc.add_table(rows=4, cols=5)
    table_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_ds.autofit = False

    headers_ds = ["Subset Data", "Kelas Banjir", "Kelas Bukan Banjir", "Total Sampel", "Proporsi (%)"]
    for j, h_text in enumerate(headers_ds):
        cell = table_ds.cell(0, j)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(h_text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)

    data_ds = [
        ["Data Latih (Train Set)", "2.000", "2.000", "4.000", "64.72%"],
        ["Data Validasi (Val Set)", "500", "500", "1.000", "16.18%"],
        ["Data Uji (Test Set)", "552", "628", "1.180", "19.10%"],
    ]
    for i, row_data in enumerate(data_ds):
        row_bg = "F2F5F8" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row_data):
            cell = table_ds.cell(i + 1, j)
            set_cell_background(cell, row_bg)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(val)
            run.font.size = Pt(10.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("B. Konfigurasi Pelatihan dan Augmentasi")
    r2.font.size = Pt(12)
    r2.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Pelatihan model menggunakan optimizer AdamW dengan bobot pembobotan regularisasi L2 (weight decay = 1e-4), "
        "learning rate awal 1e-4 dengan learning rate scheduler, batch size 32, serta fungsi kerugian Cross-Entropy Loss. "
        "Augmentasi citra diaplikasikan pada subset pelatihan meliputi Random Horizontal Flip (p=0.5), "
        "Random Rotation (±15°), Color Jitter, dan Random Affine."
    )

    # 3. HASIL EVALUASI
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    r1 = h1.add_run("3. Hasil Pengujian dan Evaluasi Kinerja (BAB IV Hasil dan Pembahasan)")
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x11, 0x33, 0x66)

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Evaluasi model dilakukan pada 1.180 citra data uji independen yang belum pernah dilihat sama sekali oleh model. "
        "Metrik yang dievaluasi mencakup Akurasi (Accuracy), Presisi (Precision), Sensitivitas (Recall), dan F1-Score."
    )

    # Tabel Classification Report
    table_rep = doc.add_table(rows=6, cols=5)
    table_rep.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_rep.autofit = False

    headers_rep = ["Kelas / Metrik", "Precision", "Recall", "F1-Score", "Support (Sampel)"]
    for j, h_text in enumerate(headers_rep):
        cell = table_rep.cell(0, j)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(h_text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)

    data_rep = [
        ["Banjir", "0.9277 (92.77%)", "0.9529 (95.29%)", "0.9401 (94.01%)", "552"],
        ["Bukan Banjir", "0.9576 (95.76%)", "0.9347 (93.47%)", "0.9460 (94.60%)", "628"],
        ["Akurasi Keseluruhan", "-", "-", "-", "0.9432 (94.32%)"],
        ["Macro Average", "0.9426 (94.26%)", "0.9438 (94.38%)", "0.9431 (94.31%)", "1.180"],
        ["Weighted Average", "0.9436 (94.36%)", "0.9432 (94.32%)", "0.9433 (94.33%)", "1.180"]
    ]
    for i, row_data in enumerate(data_rep):
        row_bg = "D9E1F2" if i == 2 else ("F2F5F8" if i % 2 == 1 else "FFFFFF")
        for j, val in enumerate(row_data):
            cell = table_rep.cell(i + 1, j)
            set_cell_background(cell, row_bg)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(val)
            if i in [0, 1, 2]:
                run.bold = True
            run.font.size = Pt(10.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Confusion Matrix Table & Explanation
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("A. Confusion Matrix dan Analisis Kesalahan")
    r2.font.size = Pt(12)
    r2.font.bold = True

    # Tabel Confusion Matrix
    table_cm = doc.add_table(rows=3, cols=3)
    table_cm.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_cm.autofit = False

    headers_cm = ["Aktual \\ Prediksi", "Prediksi: Banjir", "Prediksi: Bukan Banjir"]
    for j, h_text in enumerate(headers_cm):
        cell = table_cm.cell(0, j)
        set_cell_background(cell, "2E75B6")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        cp = cell.paragraphs[0]
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(h_text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10.5)

    data_cm = [
        ["Aktual: Banjir", "526 (True Positive - 95.3%)", "26 (False Negative - 4.7%)"],
        ["Aktual: Bukan Banjir", "41 (False Positive - 6.5%)", "587 (True Negative - 93.5%)"]
    ]
    for i, row_data in enumerate(data_cm):
        for j, val in enumerate(row_data):
            cell = table_cm.cell(i + 1, j)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            cp = cell.paragraphs[0]
            cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = cp.add_run(val)
            run.font.size = Pt(10.5)
            if j > 0:
                run.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p_analisis = doc.add_paragraph()
    p_analisis.paragraph_format.line_spacing = 1.15
    p_analisis.paragraph_format.space_after = Pt(6)
    p_analisis.add_run(
        "Analisis Pembahasan Skripsi / Tugas Akhir:\n"
        "1. True Positive (TP = 526): Model berhasil mengenali 526 kejadian banjir nyata secara tepat.\n"
        "2. False Negative (FN = 26): Tingkat kesalahan tipe II (kejadian banjir terlewat) hanya 4.7% (26 dari 552).\n"
        "3. False Positive (FP = 41): 41 foto bukan banjir diprediksi sebagai banjir (misalnya genangan kecil atau aspal basah).\n"
        "4. True Negative (TN = 587): 587 foto lingkungan kering teridentifikasi benar.\n"
        "5. Keunggulan Nilai Recall (95.29%): Dalam konteks sistem peringatan dini dan pelaporan bencana, "
        "nilai recall tinggi pada kelas banjir merupakan aspek paling krusial karena memastikan tidak ada bencana nyata "
        "yang diabaikan oleh sistem penanganan darurat."
    )

    # Tambahkan gambar heatmap jika file ada
    img_path = Path("models/confusion_matrix.png")
    if img_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(4.5))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run("Gambar 1. Visualisasi Heatmap Confusion Matrix Model MobileNetV3-Large")
        run_cap.font.size = Pt(10)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # 4. INTEGRASI WEBGIS
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(6)
    r1 = h1.add_run("4. Integrasi ke Sistem WebGIS MERN Stack (BAB III / BAB IV)")
    r1.font.size = Pt(14)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x11, 0x33, 0x66)

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Untuk kebutuhan integrasi pada sistem WebGIS (MERN Stack), model diekspor ke format terbuka ONNX "
        "(flood_mobilenetv3.onnx). Format ini memungkinkan inferensi mandiri yang cepat baik melalui arsitektur "
        "Microservice (FastAPI dipanggil Express.js) maupun langsung pada environment Node.js via onnxruntime-node. "
        "Logika penyortiran pada dashboard admin WebGIS dikonfigurasikan sebagai berikut:"
    )

    logics = [
        ("Kategori Prioritas Tinggi (Confidence >= 85% & Banjir): ", 
         "Laporan terverifikasi valid secara otomatis oleh AI dan langsung direkomendasikan untuk disetujui (Auto-Approved) sehingga langsung muncul pada layer peta publik WebGIS."),
        ("Kategori Perlu Tinjauan Manual (50% <= Confidence < 85% & Banjir): ", 
         "Laporan ditandai membutuhkan inspeksi visual oleh petugas admin sebelum titik koordinat diterbitkan ke peta publik."),
        ("Kategori Ditolak / Spam (Bukan Banjir): ", 
         "Laporan yang memuat foto bukan banjir (misal jalanan kering, foto salah upload, atau spam) disaring otomatis ke daftar tolak sehingga tidak membebani kerja verifikasi admin.")
    ]
    for l_title, l_desc in logics:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        bp.paragraph_format.space_after = Pt(3)
        rt = bp.add_run(l_title)
        rt.bold = True
        bp.add_run(l_desc)

    out_file = Path("DOKUMEN_METODOLOGI_MODEL_AI_TUGAS_AKHIR.docx")
    doc.save(out_file)
    print(f"[SUCCESS] Berhasil membuat dokumen Word: {out_file.resolve()}")

if __name__ == "__main__":
    create_document()
