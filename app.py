"""
app.py
====================
TAHAP 9 (Bab 11): Aplikasi Web Flask untuk Klasifikasi Jenis Bunga
menggunakan model Transfer Learning VGG16.

Rute (routes):
    /            -> Halaman Home (penjelasan singkat + tombol upload)
    /predict     -> GET: halaman upload+form, POST: proses upload & prediksi
    /result      -> Halaman hasil (gambar, nama bunga, confidence)

Validasi:
    - Hanya menerima file berekstensi jpg, jpeg, png
    - Menampilkan pesan flash jika file bukan gambar / upload gagal
"""

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash

from predict import predict_image, get_model

app = Flask(__name__)
app.secret_key = "flower-classification-secret-key"  # dipakai untuk flash message

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # maksimal 5 MB per file

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Validasi: file harus punya ekstensi jpg/jpeg/png."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    """Halaman Home: judul aplikasi, penjelasan singkat, tombol upload."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  -> tampilkan form upload gambar
    POST -> proses gambar yang diupload, validasi, lalu jalankan prediksi
    """
    if request.method == "POST":
        # Validasi: apakah ada file pada request
        if "file" not in request.files:
            flash("Tidak ada file yang dipilih.")
            return redirect(request.url)

        file = request.files["file"]

        # Validasi: apakah user benar-benar memilih file
        if file.filename == "":
            flash("Tidak ada file yang dipilih.")
            return redirect(request.url)

        # Validasi: apakah ekstensi file diizinkan (jpg/jpeg/png)
        if not allowed_file(file.filename):
            flash("Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.")
            return redirect(request.url)

        try:
            # Simpan file dengan nama unik agar tidak bentrok antar user
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)
        except Exception:
            flash("Upload gagal. Silakan coba lagi.")
            return redirect(request.url)

        try:
            # Jalankan prediksi menggunakan model VGG16 (predict.py)
            label, confidence = predict_image(filepath)
        except Exception:
            flash("Terjadi kesalahan saat memproses gambar. Silakan coba lagi.")
            return redirect(request.url)

        return render_template(
            "result.html",
            image_path=filepath.replace("\\", "/"),
            label=label.capitalize(),
            confidence=f"{confidence:.2f}",
        )

    return render_template("predict.html")


if __name__ == "__main__":
    # Load model sekali di awal agar prediksi pertama tidak lambat
    get_model()
    app.run(debug=True)
