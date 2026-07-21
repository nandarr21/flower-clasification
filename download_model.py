import os
import urllib.request

MODEL_URL = (
    "https://huggingface.co/nandar21/flower-vgg16-model/"
    "resolve/main/model_vgg16_flower.keras"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model_vgg16_flower.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)


if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
    print(f"Model sudah tersedia: {MODEL_PATH}")

else:
    print("Model belum tersedia.")
    print("Mengunduh model dari Hugging Face...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model berhasil diunduh.")
    print(f"Lokasi model: {MODEL_PATH}")
    print(
        f"Ukuran model: "
        f"{os.path.getsize(MODEL_PATH)} bytes"
    )