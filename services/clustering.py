"""
services/clustering.py
======================
Modul untuk menjalankan algoritma klasterisasi:
- K-Means: mencari kelompok berdasarkan jarak ke centroid
- DBSCAN : mendeteksi klaster berbasis densitas (kepadatan titik)
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------
# Fungsi: Jalankan K-Means untuk K = 2 sampai 10
# -----------------------------------------------------------
def run_kmeans_analysis(X_scaled):
    """
    Menjalankan K-Means untuk berbagai nilai K (2–10),
    lalu menghitung Inertia (WCSS) dan Silhouette Score.

    Args:
        X_scaled (ndarray): Data yang sudah di-scaling.

    Returns:
        hasil (list of dict): Setiap elemen berisi:
            - k         : Jumlah klaster
            - inertia   : Within-Cluster Sum of Squares
            - silhouette: Silhouette Score (-1 hingga 1)
            - labels    : Label klaster tiap sampel
        k_terbaik (int): K dengan Silhouette Score tertinggi.
    """
    hasil = []
    k_range = range(2, 11)  # K dari 2 sampai 10

    for k in k_range:
        # Inisialisasi dan jalankan K-Means
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,   # Hasil reproduktif
            n_init=10          # Jumlah inisialisasi berbeda
        )
        labels = kmeans.fit_predict(X_scaled)

        # Hitung Silhouette Score
        # Nilai mendekati 1 = klaster terpisah dengan baik
        sil = silhouette_score(X_scaled, labels)

        hasil.append({
            "k": k,
            "inertia": round(float(kmeans.inertia_), 2),
            "silhouette": round(float(sil), 4),
            "labels": labels.tolist(),
        })

    # Temukan K dengan Silhouette Score tertinggi
    k_terbaik = max(hasil, key=lambda x: x["silhouette"])["k"]

    return hasil, k_terbaik


# -----------------------------------------------------------
# Fungsi: Hitung k-distance graph untuk DBSCAN
# -----------------------------------------------------------
def compute_k_distance(X_scaled, k=4):
    """
    Menghitung jarak ke tetangga ke-k untuk setiap titik,
    diurutkan secara menaik. Grafik ini digunakan untuk
    menentukan nilai eps pada DBSCAN (cari 'lutut' grafiknya).

    Args:
        X_scaled (ndarray): Data yang sudah di-scaling.
        k (int)           : Urutan tetangga terdekat.

    Returns:
        distances_sorted (ndarray): Jarak yang sudah diurutkan menaik.
    """
    # Cari k+1 tetangga terdekat (termasuk titik itu sendiri)
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(X_scaled)

    # Ambil jarak ke tetangga ke-k (bukan tetangga ke-0 = dirinya sendiri)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = distances[:, k]  # Ambil kolom ke-k

    # Urutkan dari kecil ke besar
    distances_sorted = np.sort(k_distances)

    return distances_sorted


# -----------------------------------------------------------
# Fungsi: Jalankan DBSCAN dengan satu kombinasi parameter
# -----------------------------------------------------------
def run_dbscan(X_scaled, eps, min_samples):
    """
    Menjalankan DBSCAN dengan parameter yang diberikan.

    Args:
        X_scaled    (ndarray): Data yang sudah di-scaling.
        eps         (float)  : Radius lingkungan (epsilon).
        min_samples (int)    : Jumlah minimum titik dalam lingkungan.

    Returns:
        dict berisi:
            - labels       : Label klaster tiap sampel (-1 = noise)
            - n_clusters   : Jumlah klaster (tanpa noise)
            - n_noise      : Jumlah titik noise (label = -1)
            - eps          : Parameter eps yang digunakan
            - min_samples  : Parameter min_samples yang digunakan
            - error        : Pesan error (None jika sukses)
    """
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # Hitung jumlah klaster (abaikan label -1 yang berarti noise)
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})  # Hapus -1 dari hitungan
        n_noise = int(np.sum(labels == -1))

        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "eps": eps,
            "min_samples": min_samples,
            "error": None,
        }

    except Exception as e:
        return {
            "labels": [],
            "n_clusters": 0,
            "n_noise": 0,
            "eps": eps,
            "min_samples": min_samples,
            "error": str(e),
        }


# -----------------------------------------------------------
# Fungsi: Jalankan beberapa kombinasi DBSCAN
# -----------------------------------------------------------
def run_dbscan_combinations(X_scaled, eps_values, min_samples_values):
    """
    Menjalankan DBSCAN untuk beberapa kombinasi eps dan min_samples.

    Args:
        X_scaled         (ndarray): Data yang sudah di-scaling.
        eps_values       (list)   : Daftar nilai eps.
        min_samples_values (list) : Daftar nilai min_samples (harus sama panjang).

    Returns:
        list of dict: Hasil setiap kombinasi.
    """
    hasil = []
    for eps, ms in zip(eps_values, min_samples_values):
        res = run_dbscan(X_scaled, eps=eps, min_samples=ms)
        hasil.append(res)
    return hasil


# -----------------------------------------------------------
# Fungsi: Estimasi eps dari k-distance graph
# -----------------------------------------------------------
def estimate_eps_from_kdistance(distances_sorted):
    """
    Memperkirakan nilai eps dari k-distance graph dengan
    mencari titik di mana kemiringan (slope) paling besar berubah
    (metode 'knee detection' sederhana).

    Returns:
        eps_knee  (float): Estimasi eps di titik lutut.
        eps_lower (float): Eps sedikit lebih kecil dari lutut.
        eps_upper (float): Eps sedikit lebih besar dari lutut.
    """
    n = len(distances_sorted)

    # Hitung perubahan slope (aproksimasi turunan kedua)
    # Cari titik dengan perubahan paling tajam
    if n < 10:
        # Data terlalu sedikit, pakai median
        eps_knee = float(np.median(distances_sorted))
    else:
        # Gunakan metode sederhana: cari kemiringan maksimum
        # antara titik 20% - 80% data
        start_idx = n // 5
        end_idx = 4 * n // 5

        slopes = np.diff(distances_sorted[start_idx:end_idx])
        idx_max_slope = np.argmax(slopes) + start_idx
        eps_knee = float(distances_sorted[idx_max_slope])

    # Buat variasi eps
    eps_lower = round(max(0.1, eps_knee * 0.6), 2)
    eps_knee_rounded = round(eps_knee, 2)
    eps_upper = round(eps_knee * 1.5, 2)

    return eps_knee_rounded, eps_lower, eps_upper
