

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # agar bisa jalan tanpa display (server/headless)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model_vgg16_flower.keras")

EPOCHS_INITIAL = 15       # training tahap 1 (feature extraction)
EPOCHS_FINE_TUNE = 10     # training tahap 2 (fine tuning)
FINE_TUNE_AT_LAYER = 15   # buka layer VGG16 mulai dari index ini (dari total 19 layer)
FINE_TUNE_LR = 1e-5       # learning rate kecil untuk fine tuning

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------------
# TAHAP 2: PREPROCESSING - ImageDataGenerator
# ------------------------------------------------------------------
def build_generators():
    """
    Training generator diberi augmentasi (rotasi, zoom, shear, flip)
    agar model lebih general dan tidak overfit pada dataset kecil.
    Validation & test generator HANYA di-rescale (tanpa augmentasi),
    karena keduanya dipakai untuk mengukur performa model secara jujur/apa
    adanya, bukan untuk melatih.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_data = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_data, val_data, test_data


# ------------------------------------------------------------------
# TAHAP 3: MEMBANGUN MODEL VGG16 (TRANSFER LEARNING)
# ------------------------------------------------------------------
def build_model():
    """
    Load VGG16 tanpa Fully Connected layer (include_top=False), lalu
    bekukan (freeze) seluruh layer-nya agar bobot hasil training ImageNet
    tidak rusak pada tahap awal. Di atasnya ditambahkan classifier head
    baru sesuai Bab 11: Flatten -> Dense(256, relu) -> Dropout(0.5) ->
    Dense(5, softmax) untuk 5 kelas bunga.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze seluruh layer VGG16 (feature extractor tidak ikut dilatih dulu)
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model, base_model


# ------------------------------------------------------------------
# TAHAP 4: TRAINING (feature extraction, base model masih dibekukan)
# ------------------------------------------------------------------
def train_initial(model, train_data, val_data):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks,
    )
    return history


# ------------------------------------------------------------------
# TAHAP 5: FINE TUNING
# ------------------------------------------------------------------
def fine_tune(model, base_model, train_data, val_data):
    """
    Fine Tuning membuka sebagian layer terakhir VGG16 agar ikut belajar
    menyesuaikan diri dengan karakteristik khusus dataset bunga (tekstur
    kelopak, warna, bentuk daun), bukan hanya fitur umum dari ImageNet.

    Alasan setiap langkah:
    - Hanya membuka layer TERAKHIR (bukan semua layer): layer awal VGG16
      menangkap fitur generik (tepi, warna, tekstur) yang tetap berguna
      untuk gambar apa pun, sehingga tidak perlu dilatih ulang. Layer
      akhir menangkap fitur yang lebih spesifik/abstrak, sehingga paling
      diuntungkan bila disesuaikan dengan dataset baru.
    - Learning rate dibuat SANGAT KECIL (1e-5): karena bobot VGG16 sudah
      bagus (hasil training ImageNet), kita hanya ingin melakukan
      penyesuaian halus. Learning rate besar berisiko merusak
      (catastrophic forgetting) fitur yang sudah dipelajari dengan baik.
    - Model di-compile ULANG setelah mengubah status trainable, karena
      Keras hanya mendaftarkan variabel yang bisa dilatih pada saat
      compile dipanggil.
    """
    base_model.trainable = True

    # Bekukan kembali layer-layer awal, hanya buka beberapa layer terakhir
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False
    for layer in base_model.layers[FINE_TUNE_AT_LAYER:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )
    return history_fine


# ------------------------------------------------------------------
# TAHAP 6: EVALUASI MODEL
# ------------------------------------------------------------------
def evaluate_model(model, test_data):
    test_loss, test_acc = model.evaluate(test_data)
    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}\n")

    # Prediksi seluruh data test untuk classification report & confusion matrix
    test_data.reset()
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_data.classes

    print("=== Classification Report ===")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    print("=== Confusion Matrix ===")
    print(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - Klasifikasi Bunga (VGG16)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    return test_loss, test_acc


def plot_history(history_initial, history_fine):
    """
    Menggabungkan riwayat training tahap awal (feature extraction) dan
    fine tuning menjadi satu grafik yang berkesinambungan, agar terlihat
    jelas titik dimulainya fine tuning.
    """
    acc = history_initial.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_initial.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history_initial.history["loss"] + history_fine.history["loss"]
    val_loss = history_initial.history["val_loss"] + history_fine.history["val_loss"]

    fine_tune_start = len(history_initial.history["accuracy"])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(fine_tune_start, color="gray", linestyle="--", label="Mulai Fine Tuning")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(fine_tune_start, color="gray", linestyle="--", label="Mulai Fine Tuning")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(">> Menyiapkan data generator...")
    train_data, val_data, test_data = build_generators()

    print(">> Membangun model VGG16 (transfer learning)...")
    model, base_model = build_model()

    print(">> Training tahap 1 (feature extraction)...")
    history_initial = train_initial(model, train_data, val_data)

    print(">> Fine tuning model...")
    history_fine = fine_tune(model, base_model, train_data, val_data)

    print(">> Evaluasi model pada data test...")
    evaluate_model(model, test_data)

    print(">> Membuat grafik riwayat training...")
    plot_history(history_initial, history_fine)

    print(f">> Menyimpan model final ke {MODEL_PATH} ...")
    model.save(MODEL_PATH)

    print("\nSelesai. Model, grafik, classification report, dan confusion matrix telah dibuat.")
