# Klasifikasi Jenis Bunga dengan Transfer Learning VGG16 (Flask)

Aplikasi web untuk mengklasifikasikan jenis bunga (Daisy, Dandelion, Rose,
Sunflower, Tulip) menggunakan **Transfer Learning VGG16**.

## Struktur Project

```
flower-classification/
│
├── app.py                 # Aplikasi web Flask
├── train_model.py         # Preprocessing, training, fine tuning, evaluasi
├── predict.py              # Fungsi & CLI untuk prediksi satu gambar
├── prepare_dataset.py      # Membagi dataset mentah menjadi train/val/test
├── requirements.txt
│
├── model/
│   └── model_vgg16_flower.keras   # Model hasil training (dibuat setelah training)
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── templates/
│   ├── base.html
│   ├── index.html          # Halaman Home
│   ├── predict.html        # Halaman Prediksi (upload gambar)
│   └── result.html         # Halaman Hasil
│
├── static/
│   ├── css/style.css
│   └── uploads/             # Tempat menyimpan gambar yang diupload user
│
└── README.md
```

## Cara Menjalankan Secara Lokal

### 1. Buat virtual environment & install dependency

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan dataset

1. Download dataset **Flowers Recognition** dari Kaggle:
   https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
2. Ekstrak ke folder `raw_dataset/` di root project (folder berisi 5
   subfolder: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`).
3. Jalankan:

   ```bash
   python prepare_dataset.py
   ```

   Script ini akan membagi dataset menjadi `dataset/train`,
   `dataset/validation`, dan `dataset/test` secara otomatis (70/15/15).

### 3. Training model (termasuk Fine Tuning)

```bash
python train_model.py
```

Proses ini akan:
- Melakukan training tahap awal (feature extraction, VGG16 dibekukan)
- Melakukan fine tuning (membuka beberapa layer terakhir VGG16 dengan
  learning rate kecil)
- Mengevaluasi model pada data test (accuracy, loss, classification
  report, confusion matrix)
- Menyimpan grafik `training_history.png` dan `confusion_matrix.png`
- Menyimpan model akhir ke `model/model_vgg16_flower.keras`

> Training membutuhkan waktu cukup lama tanpa GPU. Disarankan menjalankan
> di lingkungan dengan GPU (Google Colab, Kaggle Notebook, atau komputer
> lokal dengan GPU) untuk mempercepat proses.

### 4. (Opsional) Uji prediksi lewat command line

```bash
python predict.py path/ke/gambar_bunga.jpg
```

### 5. Jalankan aplikasi web Flask

```bash
python app.py
```

Buka browser ke `http://127.0.0.1:5000`.

## Alur Aplikasi

1. **Halaman Home** — penjelasan singkat aplikasi + tombol menuju halaman prediksi.
2. **Halaman Prediksi** — pengguna mengunggah gambar (JPG/JPEG/PNG), melihat
   preview, lalu menekan tombol "Prediksi".
3. **Halaman Hasil** — menampilkan gambar yang diunggah, nama bunga hasil
   klasifikasi, dan nilai confidence (%) dalam bentuk progress bar.

## Validasi

- Hanya menerima file dengan ekstensi `.jpg`, `.jpeg`, `.png`.
- Menampilkan pesan peringatan (flash message) jika file bukan gambar
  yang didukung, atau jika proses upload/prediksi gagal.

## Deployment

Aplikasi ini bisa dijalankan langsung secara lokal (`python app.py`) atau
di-deploy ke layanan hosting yang mendukung Python/Flask (misalnya Render
atau Railway):

1. Push seluruh project ke GitHub (**kecualikan folder `dataset/` dan
   `raw_dataset/`** dari repo — cukup sertakan `model/model_vgg16_flower.keras`
   hasil training).
2. Di platform hosting, set start command: `gunicorn app:app` (tambahkan
   `gunicorn` ke `requirements.txt` bila dipakai).
3. Pastikan environment memiliki cukup memori (model VGG16 relatif besar,
   ~500MB+), pilih paket hosting yang sesuai.

## Teknologi yang Digunakan

- **TensorFlow / Keras** — VGG16 pretrained model, transfer learning, fine tuning
- **Flask** — web framework backend
- **Bootstrap 5** — tampilan responsif (navbar, hero section, card, footer)
- **scikit-learn** — classification report & confusion matrix
- **Matplotlib / Seaborn** — visualisasi grafik training & confusion matrix
