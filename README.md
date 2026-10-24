# Unsupervised Learning Explorer 🔬

Aplikasi web Flask interaktif untuk tugas kuliah **Machine Learning — Unsupervised Learning**.

## Deskripsi Kasus

Dataset yang digunakan adalah **Breast Cancer Wisconsin (Diagnostic)** dari UCI Machine Learning Repository, tersedia langsung melalui `sklearn.datasets.load_breast_cancer`.

Dataset ini berisi hasil komputasi citra digital dari aspirasi jarum halus (FNA) benjolan payudara pasien. Setiap sampel memiliki:
- **569 sampel** (212 ganas / 357 jinak)
- **30 fitur numerik** (radius, tekstur, perimeter, area, kehalusan, dll.)
- **Label asli** (malignant / benign) — hanya digunakan sebagai pembanding, **tidak dipakai saat klasterisasi**

**Tujuan**: Menemukan pola atau kelompok alami dalam data tanpa menggunakan label — seolah-olah kita tidak tahu diagnosis setiap pasien.

---

## Metode yang Diimplementasikan

| Kategori | Metode |
|---|---|
| Klasterisasi | K-Means (K=2–10), DBSCAN |
| Reduksi Dimensi | PCA, t-SNE, UMAP |
| Evaluasi | Silhouette Score, Inertia, Explained Variance |

---

## Struktur Proyek

```
unsupervised-app/
├── app.py                  # Aplikasi Flask utama (semua route)
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi ini
├── .gitignore
├── services/
│   ├── data.py             # Muat data, validasi CSV, scaling
│   ├── clustering.py       # K-Means, DBSCAN
│   ├── reduction.py        # PCA, t-SNE, UMAP
│   └── plotting.py         # Semua fungsi plot matplotlib
├── templates/
│   ├── base.html           # Template dasar (navbar, layout)
│   ├── beranda.html        # Halaman 1: Beranda
│   ├── preprocessing.html  # Halaman 2: Preprocessing & scaling
│   ├── kmeans.html         # Halaman 3: K-Means clustering
│   ├── dbscan.html         # Halaman 4: DBSCAN clustering
│   ├── pca.html            # Halaman 5: PCA
│   ├── tsne.html           # Halaman 6: t-SNE
│   ├── umap.html           # Halaman 7: UMAP
│   └── visualisasi.html    # Halaman 8: Visualisasi gabungan
├── static/
│   └── plots/              # File PNG plot yang digenerate
└── uploads/                # File CSV sementara (di .gitignore)
```

---

## Cara Instalasi & Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/<username>/unsupervised-app.git
cd unsupervised-app
```

### 2. Buat virtual environment (disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

> **Catatan**: `umap-learn` membutuhkan beberapa detik untuk mengkompilasi. Jika gagal, aplikasi tetap berjalan — halaman UMAP akan menampilkan pesan error yang jelas.

### 4. Jalankan aplikasi

```bash
python app.py
```

Buka browser dan akses: **http://localhost:5000**

---

## Cara Penggunaan

### Dengan Dataset Bawaan (Default)
1. Buka http://localhost:5000
2. Klik **Mulai Analisis** atau navigasi ke **Preprocessing**
3. Dataset Breast Cancer akan dimuat otomatis
4. Jelajahi setiap halaman dari kiri ke kanan (Preprocessing → K-Means → DBSCAN → PCA → t-SNE → UMAP → Gabungan)

### Dengan CSV Sendiri
1. Buka halaman **Preprocessing**
2. Pilih **"Unggah File CSV Sendiri"**
3. Unggah file (maks. 5 MB, format .csv)
4. (Opsional) Masukkan nama kolom label jika ada
5. Klik **Muat Data**

**Syarat CSV**:
- Minimal 3 kolom numerik
- Tidak boleh ada missing values
- Ukuran ≤ 5 MB

---

## Catatan Teknis

- **Cache**: Hasil t-SNE dan UMAP disimpan di memori agar tidak dihitung ulang saat halaman dibuka kembali. Cache dihapus otomatis jika sumber data diganti.
- **Backend Matplotlib**: Menggunakan `Agg` (non-GUI) agar berjalan di server tanpa display.
- **Random State**: Semua metode acak menggunakan `random_state=42` untuk reproduktifitas.
- **DBSCAN 0 Klaster**: Ditangani dengan pesan peringatan yang jelas, aplikasi tidak error.
- **UMAP Tidak Tersedia**: Ditangani dengan graceful degradation — halaman lain tetap berfungsi.

---

## Dependensi

| Paket | Versi Min | Fungsi |
|---|---|---|
| Flask | >=3.0.0 | Framework web |
| scikit-learn | >=1.5.0 | Algoritma ML (max_iter TSNE butuh >=1.5) |
| pandas | >=2.1.0 | Manipulasi data |
| numpy | >=1.26.0 | Komputasi numerik |
| matplotlib | >=3.8.0 | Visualisasi |
| umap-learn | >=0.5.5 | UMAP |
| Werkzeug | >=3.0.0 | Utilitas Flask |

---

## Lisensi

Proyek ini dibuat untuk keperluan tugas kuliah. Dataset Breast Cancer Wisconsin tersedia di domain publik melalui [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
