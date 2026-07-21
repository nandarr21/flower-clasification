"""
predict.py
====================
TAHAP 8 (Bab 11): Prediksi gambar bunga menggunakan model hasil training.

Langkah prediksi:
    1. Load model (model/model_vgg16_flower.keras)
    2. Terima path gambar (upload)
    3. Resize gambar menjadi 224x224
    4. Rescale piksel (1./255)
    5. Prediksi kelas menggunakan model
    6. Ambil confidence score (nilai probabilitas tertinggi)
    7. Tampilkan nama bunga hasil prediksi

Dapat dijalankan langsung dari terminal:
    python predict.py path/ke/gambar.jpg

Atau di-import sebagai modul (dipakai oleh app.py):
    from predict import predict_image
    label, confidence = predict_image("static/uploads/gambar.jpg")
"""

import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/model_vgg16_flower.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

_model = None  # cache model agar tidak di-load berulang kali


def get_model():
    """Load model sekali saja (lazy loading), lalu simpan di cache."""
    global _model
    if _model is None:
        print(f"Loading model dari {MODEL_PATH} ...")
        _model = load_model(MODEL_PATH)
    return _model


def predict_image(img_path):
    """
    Melakukan prediksi kelas bunga dari sebuah file gambar.

    Return:
        label (str)       : nama kelas bunga dengan probabilitas tertinggi
        confidence (float) : nilai keyakinan model dalam persen (0-100)
    """
    model = get_model()

    # 1. Load gambar & resize menjadi 224x224 sesuai input VGG16
    img = image.load_img(img_path, target_size=IMG_SIZE)

    # 2. Ubah gambar menjadi array numpy
    img_array = image.img_to_array(img)

    # 3. Rescale piksel ke rentang 0-1 (sama seperti saat training)
    img_array = img_array / 255.0

    # 4. Tambahkan dimensi batch -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prediksi
    predictions = model.predict(img_array)[0]

    # 6. Ambil kelas dengan probabilitas tertinggi + confidence score
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    # 7. Nama bunga hasil prediksi
    label = CLASS_NAMES[predicted_index]

    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara pakai: python predict.py path/ke/gambar.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    label, confidence = predict_image(img_path)
    print(f"Hasil Prediksi : {label}")
    print(f"Confidence     : {confidence:.2f}%")
