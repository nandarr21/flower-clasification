import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# ============================================================
# KONFIGURASI
# ============================================================

# Ambil folder tempat file predict.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path model absolut
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model_vgg16_flower.keras"
)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


def get_model():
    """
    Load model hanya sekali.
    Model disimpan di cache agar tidak di-load berulang kali.
    """

    global _model

    if _model is None:

        print(
            f"Loading model dari: {MODEL_PATH}",
            flush=True
        )

        # Pastikan file model ada
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model tidak ditemukan di: {MODEL_PATH}"
            )

        try:

            _model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )

            print(
                "Model berhasil dimuat!",
                flush=True
            )

        except Exception as e:

            print(
                f"Gagal memuat model: {e}",
                flush=True
            )

            raise

    return _model


# ============================================================
# PREDIKSI
# ============================================================

def predict_image(img_path):
    """
    Melakukan prediksi kelas bunga dari sebuah file gambar.

    Return:
        label (str)
        confidence (float) -> persen 0-100
    """

    # Pastikan file gambar ada
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {img_path}"
        )

    # Ambil model
    model = get_model()

    # --------------------------------------------------------
    # 1. Load gambar
    # --------------------------------------------------------

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # --------------------------------------------------------
    # 2. Convert gambar ke numpy array
    # --------------------------------------------------------

    img_array = image.img_to_array(img)

    # --------------------------------------------------------
    # 3. Normalisasi
    # --------------------------------------------------------

    img_array = img_array / 255.0

    # --------------------------------------------------------
    # 4. Tambahkan batch dimension
    # Shape:
    # (224, 224, 3)
    # menjadi:
    # (1, 224, 224, 3)
    # --------------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # 5. Prediksi
    # --------------------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # 6. Ambil index probabilitas terbesar
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    # --------------------------------------------------------
    # 7. Confidence
    # --------------------------------------------------------

    confidence = (
        float(predictions[predicted_index])
        * 100
    )

    # --------------------------------------------------------
    # 8. Nama kelas
    # --------------------------------------------------------

    label = CLASS_NAMES[
        predicted_index
    ]

    return label, confidence


# ============================================================
# TEST VIA TERMINAL
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Cara pakai:"
        )

        print(
            "python predict.py path/ke/gambar.jpg"
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