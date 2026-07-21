import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from predict import predict_image


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "flower-classification-secret-key"


# ============================================================
# PATH KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024
)


# Pastikan folder upload tersedia
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# VALIDASI FILE
# ============================================================

def allowed_file(filename):
    """
    Validasi file:
    hanya menerima JPG, JPEG, dan PNG.
    """

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDIKSI
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "predict.html"
        )


    # --------------------------------------------------------
    # POST - VALIDASI FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        flash(
            "Tidak ada file yang dipilih."
        )

        return redirect(
            request.url
        )


    file = request.files["file"]


    if file.filename == "":

        flash(
            "Tidak ada file yang dipilih."
        )

        return redirect(
            request.url
        )


    if not allowed_file(
        file.filename
    ):

        flash(
            "Format file tidak didukung. "
            "Gunakan JPG, JPEG, atau PNG."
        )

        return redirect(
            request.url
        )


    # --------------------------------------------------------
    # SIMPAN FILE
    # --------------------------------------------------------

    try:

        ext = file.filename.rsplit(
            ".",
            1
        )[1].lower()


        unique_filename = (
            f"{uuid.uuid4().hex}.{ext}"
        )


        filepath = os.path.join(
            app.config[
                "UPLOAD_FOLDER"
            ],
            unique_filename
        )


        file.save(
            filepath
        )


        app.logger.info(
            f"File berhasil disimpan: "
            f"{filepath}"
        )


    except Exception as e:

        app.logger.exception(
            "Error saat menyimpan file:"
        )


        flash(
            f"Upload gagal: {str(e)}"
        )


        return redirect(
            request.url
        )


    # --------------------------------------------------------
    # PREDIKSI MODEL
    # --------------------------------------------------------

    try:

        app.logger.info(
            "Memulai proses prediksi..."
        )


        label, confidence = (
            predict_image(
                filepath
            )
        )


        app.logger.info(
            f"Prediksi berhasil: "
            f"{label} "
            f"({confidence:.2f}%)"
        )


    except Exception as e:

        app.logger.exception(
            "Error saat melakukan prediksi:"
        )


        flash(
            f"Terjadi kesalahan saat "
            f"memproses gambar: {str(e)}"
        )


        return redirect(
            request.url
        )


    # --------------------------------------------------------
    # HASIL
    # --------------------------------------------------------

    image_path = filepath.replace(
        "\\",
        "/"
    )


    return render_template(
        "result.html",

        image_path=image_path,

        label=label.capitalize(),

        confidence=f"{confidence:.2f}"
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port
    )