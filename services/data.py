"""
services/data.py
================
Modul untuk memuat dataset, memvalidasi file unggahan,
memeriksa kualitas data, dan melakukan scaling fitur.
"""

import os
import json
import uuid
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

# -----------------------------------------------------------
# Konstanta
# -----------------------------------------------------------
UPLOAD_FOLDER = "uploads"          # Folder penyimpanan file sementara
MAX_FILE_SIZE = 5 * 1024 * 1024    # Batas ukuran 5 MB dalam byte
ALLOWED_EXTENSIONS = {"csv"}       # Ekstensi yang diizinkan

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------------------------------------
# Fungsi: Muat dataset bawaan Breast Cancer
# -----------------------------------------------------------
def load_default_dataset():
    """
    Memuat dataset Breast Cancer Wisconsin dari scikit-learn.

    Returns:
        df_features (DataFrame): 30 fitur numerik untuk analisis.
        df_labels   (Series)   : Label asli (0=malignant, 1=benign).
        feature_names (list)   : Nama-nama fitur.
    """
    data = load_breast_cancer()

    # Buat DataFrame dari fitur-fitur (30 kolom)
    df_features = pd.DataFrame(data.data, columns=data.feature_names)

    # Label asli: 0 = malignant (ganas), 1 = benign (jinak)
    df_labels = pd.Series(data.target, name="target")

    return df_features, df_labels, list(data.feature_names)


# -----------------------------------------------------------
# Fungsi: Validasi dan muat file CSV yang diunggah
# -----------------------------------------------------------
def load_uploaded_csv(file_storage, label_column=None):
    """
    Memvalidasi dan memuat file CSV yang diunggah pengguna.

    Args:
        file_storage  : Objek file dari Flask (request.files).
        label_column  : Nama kolom yang ditandai sebagai label (opsional).

    Returns:
        dict dengan kunci:
            - 'df_features'   : DataFrame fitur numerik
            - 'df_labels'     : Series label (None jika tidak ada)
            - 'feature_names' : List nama fitur
            - 'warnings'      : List peringatan (misal kolom non-numerik)
            - 'error'         : String pesan error (None jika sukses)
    """
    hasil = {
        "df_features": None,
        "df_labels": None,
        "feature_names": [],
        "warnings": [],
        "error": None,
    }

    # Cek apakah file ada
    if file_storage is None or file_storage.filename == "":
        hasil["error"] = "Tidak ada file yang dipilih."
        return hasil

    # Cek ekstensi file
    filename = secure_filename(file_storage.filename)
    ekstensi = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ekstensi not in ALLOWED_EXTENSIONS:
        hasil["error"] = f"Format file tidak didukung: .{ekstensi}. Hanya file .csv yang diterima."
        return hasil

    # Cek ukuran file (baca ke memori, lalu cek panjangnya)
    file_storage.seek(0, 2)  # Pindah ke akhir file
    ukuran = file_storage.tell()
    file_storage.seek(0)     # Kembali ke awal
    if ukuran > MAX_FILE_SIZE:
        hasil["error"] = f"Ukuran file ({ukuran / 1024 / 1024:.1f} MB) melebihi batas 5 MB."
        return hasil

    # Baca CSV
    try:
        df = pd.read_csv(file_storage)
    except Exception as e:
        hasil["error"] = f"Gagal membaca file CSV: {str(e)}"
        return hasil

    # Cek apakah file kosong
    if df.empty or len(df) == 0:
        hasil["error"] = "File CSV kosong atau tidak memiliki baris data."
        return hasil

    # Pisahkan kolom label jika ditentukan
    df_labels = None
    if label_column and label_column in df.columns:
        df_labels = df[label_column].copy()
        df = df.drop(columns=[label_column])

    # Identifikasi kolom numerik vs non-numerik
    kolom_numerik = df.select_dtypes(include=[np.number]).columns.tolist()
    kolom_non_numerik = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Beri peringatan jika ada kolom non-numerik yang diabaikan
    if kolom_non_numerik:
        hasil["warnings"].append(
            f"Kolom berikut diabaikan karena bukan numerik: {', '.join(kolom_non_numerik)}"
        )

    # Cek minimal 3 kolom numerik
    if len(kolom_numerik) < 3:
        hasil["error"] = (
            f"Data hanya memiliki {len(kolom_numerik)} kolom numerik. "
            "Minimal 3 kolom numerik diperlukan untuk analisis."
        )
        return hasil

    # Ambil hanya kolom numerik
    df_fitur = df[kolom_numerik].copy()

    # Cek missing values
    jumlah_missing = df_fitur.isnull().sum().sum()
    if jumlah_missing > 0:
        detail_missing = df_fitur.isnull().sum()
        detail_missing = detail_missing[detail_missing > 0].to_dict()
        hasil["error"] = (
            f"Data memiliki {jumlah_missing} nilai yang hilang (missing values). "
            f"Detail per kolom: {detail_missing}. "
            "Harap bersihkan data terlebih dahulu."
        )
        return hasil

    # Semua validasi lolos
    hasil["df_features"] = df_fitur
    hasil["df_labels"] = df_labels
    hasil["feature_names"] = kolom_numerik

    return hasil


# -----------------------------------------------------------
# Fungsi: Simpan dataset aktif ke file sementara
# -----------------------------------------------------------
def save_active_data(df_features, df_labels, feature_names, upload_folder=UPLOAD_FOLDER):
    """
    Menyimpan dataset aktif ke file CSV sementara di folder uploads/.
    Ini digunakan agar data bisa diakses di semua halaman tanpa
    menyimpannya di cookie sesi (ukuran cookie terbatas).

    Returns:
        data_id (str): ID unik yang disimpan di sesi Flask.
    """
    data_id = str(uuid.uuid4())  # Buat ID unik

    # Simpan fitur
    path_fitur = os.path.join(upload_folder, f"{data_id}_features.csv")
    df_features.to_csv(path_fitur, index=False)

    # Simpan label (jika ada)
    if df_labels is not None:
        path_label = os.path.join(upload_folder, f"{data_id}_labels.csv")
        df_labels.to_frame().to_csv(path_label, index=False)

    # Simpan metadata (nama fitur, dll.)
    metadata = {
        "feature_names": feature_names,
        "has_labels": df_labels is not None,
        "label_name": df_labels.name if df_labels is not None else None,
        "n_samples": len(df_features),
        "n_features": len(feature_names),
    }
    path_meta = os.path.join(upload_folder, f"{data_id}_meta.json")
    with open(path_meta, "w") as f:
        json.dump(metadata, f)

    return data_id


# -----------------------------------------------------------
# Fungsi: Muat dataset aktif dari file sementara
# -----------------------------------------------------------
def load_active_data(data_id, upload_folder=UPLOAD_FOLDER):
    """
    Memuat dataset aktif berdasarkan data_id yang tersimpan di sesi.

    Returns:
        tuple: (df_features, df_labels, metadata) atau (None, None, None) jika tidak ditemukan.
    """
    path_fitur = os.path.join(upload_folder, f"{data_id}_features.csv")
    path_meta = os.path.join(upload_folder, f"{data_id}_meta.json")

    # Cek apakah file ada
    if not os.path.exists(path_fitur) or not os.path.exists(path_meta):
        return None, None, None

    # Muat fitur
    df_features = pd.read_csv(path_fitur)

    # Muat metadata
    with open(path_meta, "r") as f:
        metadata = json.load(f)

    # Muat label jika ada
    df_labels = None
    if metadata.get("has_labels"):
        path_label = os.path.join(upload_folder, f"{data_id}_labels.csv")
        if os.path.exists(path_label):
            df_temp = pd.read_csv(path_label)
            label_col = df_temp.columns[0]
            df_labels = df_temp[label_col]

    return df_features, df_labels, metadata


# -----------------------------------------------------------
# Fungsi: Hapus file data lama saat sumber data diganti
# -----------------------------------------------------------
def delete_old_data(data_id, upload_folder=UPLOAD_FOLDER):
    """
    Menghapus file sementara yang terkait dengan data_id lama.
    Dipanggil setiap kali pengguna mengganti sumber data.
    """
    if not data_id:
        return

    untuk_dihapus = [
        os.path.join(upload_folder, f"{data_id}_features.csv"),
        os.path.join(upload_folder, f"{data_id}_labels.csv"),
        os.path.join(upload_folder, f"{data_id}_meta.json"),
    ]

    for path in untuk_dihapus:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass  # Abaikan error penghapusan


# -----------------------------------------------------------
# Fungsi: Terapkan StandardScaler pada data fitur
# -----------------------------------------------------------
def scale_features(df_features):
    """
    Menerapkan StandardScaler pada semua fitur numerik.
    StandardScaler: mean=0, std=1 (z-score normalization).

    Returns:
        X_scaled (ndarray) : Data setelah di-scaling.
        scaler   (object)  : Objek scaler (untuk referensi).
        stats_before (dict): Statistik mean & std sebelum scaling.
        stats_after  (dict): Statistik mean & std sesudah scaling.
    """
    X = df_features.values  # Konversi ke numpy array

    # Hitung statistik sebelum scaling
    stats_before = {
        "mean": df_features.mean().round(4).to_dict(),
        "std": df_features.std().round(4).to_dict(),
    }

    # Terapkan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hitung statistik sesudah scaling
    df_scaled = pd.DataFrame(X_scaled, columns=df_features.columns)
    stats_after = {
        "mean": df_scaled.mean().round(4).to_dict(),
        "std": df_scaled.std().round(4).to_dict(),
    }

    return X_scaled, scaler, stats_before, stats_after


# -----------------------------------------------------------
# Fungsi: Analisis kualitas data
# -----------------------------------------------------------
def analyze_data_quality(df_features):
    """
    Menghitung ringkasan kualitas data:
    - Jumlah missing values per kolom
    - Jumlah baris duplikat
    - Tipe data tiap kolom

    Returns:
        dict berisi ringkasan kualitas data.
    """
    missing_per_col = df_features.isnull().sum()
    duplikat = df_features.duplicated().sum()

    # Buat tabel info kolom: nama, tipe, jumlah missing
    info_kolom = []
    for col in df_features.columns:
        info_kolom.append({
            "nama_kolom": col,
            "tipe_data": str(df_features[col].dtype),
            "missing_values": int(missing_per_col[col]),
        })

    return {
        "total_missing": int(missing_per_col.sum()),
        "total_duplikat": int(duplikat),
        "info_kolom": info_kolom,
        "n_baris": len(df_features),
        "n_kolom": len(df_features.columns),
    }
