"""
prepare_dataset.py
====================
TAHAP 1 - PERSIAPAN DATASET
Sesuai Bab 11 modul praktikum (Transfer Learning VGG16).

Tujuan:
    Mengambil dataset mentah "Flowers Recognition" dari Kaggle (5 kelas:
    daisy, dandelion, rose, sunflower, tulip) dan membaginya menjadi
    tiga folder: train, validation, dan test, sesuai struktur yang
    dibutuhkan oleh ImageDataGenerator (flow_from_directory).

Asumsi struktur dataset mentah hasil download dari Kaggle:

    raw_dataset/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/

Hasil setelah dijalankan:

    dataset/
    ├── train/
    │   ├── daisy/ ...
    │   ├── dandelion/ ...
    │   ├── rose/ ...
    │   ├── sunflower/ ...
    │   └── tulip/ ...
    ├── validation/
    │   └── (struktur sama)
    └── test/
        └── (struktur sama)

Cara pakai:
    1. Download dataset "Flowers Recognition" dari Kaggle:
       https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
    2. Ekstrak ke folder bernama "raw_dataset" di root project ini.
    3. Jalankan: python prepare_dataset.py
"""

import os
import shutil
import random

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
RAW_DATASET_DIR = "raw_dataset"      # folder hasil download Kaggle
OUTPUT_DIR = "dataset"               # folder tujuan (train/validation/test)
CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Rasio pembagian data: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42  # agar hasil split konsisten setiap dijalankan


def prepare_dataset():
    random.seed(RANDOM_SEED)

    # Validasi rasio harus berjumlah 1.0
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    assert abs(total_ratio - 1.0) < 1e-6, "Total rasio train+val+test harus 1.0"

    if not os.path.isdir(RAW_DATASET_DIR):
        raise FileNotFoundError(
            f"Folder '{RAW_DATASET_DIR}' tidak ditemukan. "
            f"Download dataset dari Kaggle terlebih dahulu dan letakkan di sini."
        )

    # Buat folder tujuan (train/validation/test) untuk setiap kelas
    for split in ["train", "validation", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    summary = {}

    for cls in CLASSES:
        class_dir = os.path.join(RAW_DATASET_DIR, cls)
        if not os.path.isdir(class_dir):
            print(f"[PERINGATAN] Folder kelas '{cls}' tidak ditemukan, dilewati.")
            continue

        # Ambil semua file gambar valid pada kelas ini
        images = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # sisanya masuk test agar tidak ada gambar yang terlewat
        n_test = n_total - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for filename in train_files:
            shutil.copy(
                os.path.join(class_dir, filename),
                os.path.join(OUTPUT_DIR, "train", cls, filename),
            )
        for filename in val_files:
            shutil.copy(
                os.path.join(class_dir, filename),
                os.path.join(OUTPUT_DIR, "validation", cls, filename),
            )
        for filename in test_files:
            shutil.copy(
                os.path.join(class_dir, filename),
                os.path.join(OUTPUT_DIR, "test", cls, filename),
            )

        summary[cls] = {
            "total": n_total,
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }

    # Cetak ringkasan hasil pembagian dataset
    print("\n=== RINGKASAN PEMBAGIAN DATASET ===")
    for cls, info in summary.items():
        print(
            f"{cls:12s} -> total: {info['total']:4d} | "
            f"train: {info['train']:4d} | "
            f"validation: {info['validation']:4d} | "
            f"test: {info['test']:4d}"
        )
    print("====================================\n")
    print("Dataset berhasil disiapkan pada folder 'dataset/'.")


if __name__ == "__main__":
    prepare_dataset()
