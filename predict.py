import os
import sys
import numpy as np
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# URL model di Hugging Face
MODEL_URL = "https://huggingface.co/nandar21/flower-vgg16-model/resolve/main/model_vgg16_flower.keras"

# Lokasi penyimpanan model di Railway
MODEL_PATH = "/tmp/model_vgg16_flower.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

_model = None


def download_model():
    """
    Download model dari Hugging Face jika belum tersedia.
    """

    if os.path.exists(MODEL_PATH):
        print("Model sudah tersedia:", MODEL_PATH)
        return

    print("Model tidak ditemukan secara lokal.")
    print("Mengunduh model dari Hugging Face...")

    try:
        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get("content-length", 0)
        )

        downloaded = 0

        with open(MODEL_PATH, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(chunk)

                    downloaded += len(chunk)

                    if total_size:
                        progress = (
                            downloaded / total_size
                        ) * 100

                        print(
                            f"Download: {progress:.1f}%",
                            end="\r"
                        )

        print("\nModel berhasil diunduh.")
        print("Lokasi:", MODEL_PATH)

    except Exception as e:

        print(
            "Gagal mengunduh model:",
            str(e)
        )

        # Hapus file jika download tidak selesai
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        raise


def get_model():

    global _model

    if _model is None:

        # Pastikan model tersedia
        download_model()

        print(
            f"Loading model dari: {MODEL_PATH}"
        )

        _model = load_model(
            MODEL_PATH,
            compile=False
        )

        print("Model berhasil dimuat!")

    return _model


def predict_image(img_path):

    model = get_model()

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {img_path}"
        )

    # Load gambar
    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # Convert ke numpy
    img_array = image.img_to_array(img)

    # Normalisasi
    img_array = img_array / 255.0

    # Tambahkan batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediksi
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # Ambil kelas terbaik
    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = (
        float(predictions[predicted_index])
        * 100
    )

    label = CLASS_NAMES[
        predicted_index
    ]

    return label, confidence


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Cara pakai:"
        )

        print(
            'python predict.py "path/ke/gambar.jpg"'
        )

        sys.exit(1)

    img_path = sys.argv[1]

    label, confidence = predict_image(
        img_path
    )

    print(
        f"Hasil Prediksi : {label}"
    )

    print(
        f"Confidence     : {confidence:.2f}%"
    )