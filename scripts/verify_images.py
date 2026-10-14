import os
from pathlib import Path
from PIL import Image

def verify_all_images(base_dir="dataset"):
    base_path = Path(base_dir)
    print("=" * 50)
    print("PEMERIKSAAN INTEGRITAS GAMBAR DATASET")
    print("=" * 50)

    total_checked = 0
    corrupt_files = []

    for split in ["train", "val", "test"]:
        split_dir = base_path / split
        if not split_dir.exists():
            continue
        for cls in ["banjir", "bukan_banjir"]:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            files = list(cls_dir.iterdir())
            print(f"Memeriksa {split}/{cls}: {len(files)} file...")
            for f in files:
                if not f.is_file():
                    continue
                total_checked += 1
                try:
                    with Image.open(f) as img:
                        img.verify()
                except Exception as e:
                    print(f"  [RUSAK] {f}: {e}")
                    corrupt_files.append(f)

    print("\n" + "=" * 50)
    print(f"Total gambar diperiksa: {total_checked}")
    print(f"Gambar rusak/invalid : {len(corrupt_files)}")
    if corrupt_files:
        print("Daftar file bermasalah:")
        for cf in corrupt_files:
            print(f" - {cf}")
    else:
        print("[SUKSES] Semua gambar valid dan dapat dibaca dengan baik!")
    print("=" * 50)

if __name__ == "__main__":
    verify_all_images()
