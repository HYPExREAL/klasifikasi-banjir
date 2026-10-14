import os
import shutil
import random
from pathlib import Path

def split_val_from_train(
    base_dir="dataset",
    val_ratio=0.20,
    seed=42
):
    base_path = Path(base_dir)
    train_path = base_path / "train"
    val_path = base_path / "val"
    test_path = base_path / "test"

    classes = ["banjir", "bukan_banjir"]
    random.seed(seed)

    print("=" * 50)
    print("VERIFIKASI & PEMBAGIAN DATASET (TRAIN -> VAL)")
    print("=" * 50)

    for cls in classes:
        cls_train_dir = train_path / cls
        cls_val_dir = val_path / cls
        cls_test_dir = test_path / cls

        cls_val_dir.mkdir(parents=True, exist_ok=True)

        # Existing files
        train_files = [f for f in cls_train_dir.iterdir() if f.is_file()]
        val_files = [f for f in cls_val_dir.iterdir() if f.is_file()]
        test_files = [f for f in cls_test_dir.iterdir() if f.is_file()] if cls_test_dir.exists() else []

        print(f"\nKelas: '{cls}'")
        print(f"  - Awal Train      : {len(train_files)} gambar")
        print(f"  - Awal Val        : {len(val_files)} gambar")
        print(f"  - Test            : {len(test_files)} gambar")

        if len(val_files) > 0:
            print(f"  [INFO] Folder val/{cls} sudah memiliki {len(val_files)} file. Lewati pemindahan.")
            continue

        num_val = int(len(train_files) * val_ratio)
        if num_val == 0:
            print(f"  [WARN] Tidak cukup data di train/{cls} untuk dipisahkan ke val.")
            continue

        # Shuffle deterministik
        random.shuffle(train_files)
        val_sample = train_files[:num_val]

        print(f"  -> Memindahkan {num_val} gambar dari train ke val ({val_ratio*100:.0f}%)...")
        for file_path in val_sample:
            dest_path = cls_val_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))

        # Re-check count
        new_train_files = [f for f in cls_train_dir.iterdir() if f.is_file()]
        new_val_files = [f for f in cls_val_dir.iterdir() if f.is_file()]
        print(f"  -> Hasil akhir:")
        print(f"     Train : {len(new_train_files)} gambar")
        print(f"     Val   : {len(new_val_files)} gambar")

    print("\n" + "=" * 50)
    print("PEMBAGIAN SELESAI!")
    print("=" * 50)

if __name__ == "__main__":
    split_val_from_train()
