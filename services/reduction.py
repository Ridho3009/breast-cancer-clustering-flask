"""
services/reduction.py
=====================
Modul untuk reduksi dimensi:
- PCA   : Principal Component Analysis (linear)
- t-SNE : t-Distributed Stochastic Neighbor Embedding (non-linear)
- UMAP  : Uniform Manifold Approximation and Projection (non-linear)
"""

import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Coba impor UMAP; tangkap semua jenis exception agar kegagalan numba
# tidak menghentikan aplikasi.
try:
    import umap
    UMAP_TERSEDIA = True
except Exception:  # meliputi ImportError, numba compile error, dll.
    UMAP_TERSEDIA = False
    umap = None

# Penanda pemanasan UMAP: dijalankan hanya sekali per proses aplikasi
_umap_warmed = False


# -----------------------------------------------------------
# Fungsi: Jalankan PCA
# -----------------------------------------------------------
def run_pca(X_scaled):
    """
    Menjalankan PCA pada data yang sudah di-scaling.
    PCA mencari arah (komponen) dengan varians maksimum.

    Args:
        X_scaled (ndarray): Data yang sudah di-scaling.

    Returns:
        dict berisi:
            - X_pca          : Proyeksi 2D (2 komponen pertama)
            - explained_ratio: Rasio varians tiap komponen
            - cumulative      : Varians kumulatif
            - n_comp_85       : Jumlah komponen untuk mencapai 85% varians
            - loadings        : Matriks loading (kontribusi fitur)
    """
    # Jalankan PCA dengan semua komponen terlebih dahulu
    # agar bisa menghitung varians kumulatif
    n_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    pca_full = PCA(n_components=n_comp, random_state=42)
    pca_full.fit(X_scaled)

    # Rasio varians tiap komponen
    explained_ratio = pca_full.explained_variance_ratio_

    # Hitung varians kumulatif
    cumulative = np.cumsum(explained_ratio)

    # Cari jumlah komponen minimum untuk mencapai 85% varians
    n_comp_85 = int(np.argmax(cumulative >= 0.85)) + 1

    # Proyeksi ke 2 komponen untuk visualisasi
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)

    return {
        "X_pca": X_pca.tolist(),
        "explained_ratio": explained_ratio.tolist(),
        "cumulative": cumulative.tolist(),
        "n_comp_85": n_comp_85,
        "var_pc1": round(float(explained_ratio[0]) * 100, 2),
        "var_pc2": round(float(explained_ratio[1]) * 100, 2),
    }


# -----------------------------------------------------------
# Fungsi: Jalankan t-SNE
# -----------------------------------------------------------
def run_tsne(X_scaled, perplexities=None):
    """
    Menjalankan t-SNE dengan beberapa nilai perplexity.
    t-SNE memetakan data ke 2 dimensi dengan mempertahankan
    hubungan lokal antar titik.

    Args:
        X_scaled    (ndarray): Data yang sudah di-scaling.
        perplexities (list)  : Daftar nilai perplexity (default: [5, 30, 50]).

    Returns:
        list of dict, masing-masing berisi:
            - perplexity  : Nilai perplexity yang digunakan
            - X_tsne      : Koordinat 2D hasil t-SNE
            - waktu_detik : Waktu eksekusi dalam detik
            - dilewati    : True jika nilai ini dilewati (sampel terlalu sedikit)
            - pesan       : Pesan penjelasan jika dilewati
    """
    if perplexities is None:
        perplexities = [5, 30, 50]

    n_samples = X_scaled.shape[0]
    hasil = []

    for perp in perplexities:
        # Perplexity harus lebih kecil dari jumlah sampel
        if perp >= n_samples:
            hasil.append({
                "perplexity": perp,
                "X_tsne": None,
                "waktu_detik": 0.0,
                "dilewati": True,
                "pesan": (
                    f"Perplexity {perp} dilewati karena harus lebih kecil "
                    f"dari jumlah sampel ({n_samples})."
                ),
            })
            continue

        # Jalankan t-SNE dan catat waktu
        mulai = time.perf_counter()
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            max_iter=1000,        # Iterasi maksimum (sklearn >= 1.5 pakai max_iter)
            learning_rate="auto",
        )
        X_tsne = tsne.fit_transform(X_scaled)
        selesai = time.perf_counter()

        hasil.append({
            "perplexity": perp,
            "X_tsne": X_tsne.tolist(),
            "waktu_detik": round(selesai - mulai, 2),
            "dilewati": False,
            "pesan": None,
        })

    return hasil


# -----------------------------------------------------------
# Fungsi: Jalankan UMAP
# -----------------------------------------------------------
def run_umap(X_scaled, n_neighbors_list=None, min_dist_list=None):
    """
    Menjalankan UMAP dengan grid kombinasi n_neighbors x min_dist.
    UMAP lebih cepat dari t-SNE dan lebih baik mempertahankan
    struktur global data.

    Args:
        X_scaled        (ndarray): Data yang sudah di-scaling.
        n_neighbors_list (list)  : Daftar nilai n_neighbors (default: [5, 15, 50]).
        min_dist_list    (list)  : Daftar nilai min_dist (default: [0.1, 0.5]).

    Returns:
        list of dict atau dict error jika UMAP tidak tersedia.
    """
    # Cek ketersediaan UMAP
    if not UMAP_TERSEDIA:
        return {
            "error": (
                "Pustaka UMAP tidak ditemukan. "
                "Instal dengan perintah: pip install umap-learn"
            )
        }

    if n_neighbors_list is None:
        n_neighbors_list = [5, 15, 50]
    if min_dist_list is None:
        min_dist_list = [0.1, 0.5]

    global _umap_warmed

    hasil = []

    # --- Pemanasan JIT numba ---
    # Kombinasi UMAP pertama sangat lambat karena numba harus mengompilasi
    # kode secara JIT. Pemanasan ini menghilangkan overhead tersebut sebelum
    # pengukuran dimulai, sehingga perbandingan waktu dengan t-SNE lebih adil.
    # Hanya dijalankan sekali per proses (bukan setiap kali halaman dibuka).
    if not _umap_warmed:
        try:
            umap.UMAP(
                n_components=2, n_neighbors=5, random_state=42
            ).fit_transform(X_scaled[:100])
        except Exception:
            pass  # Pemanasan gagal; lanjut saja, pengukuran bisa sedikit lebih lambat
        _umap_warmed = True

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            # Pastikan n_neighbors tidak melebihi jumlah sampel - 1
            n_eff = min(n_neighbors, X_scaled.shape[0] - 1)

            # Jalankan UMAP dan catat waktu
            mulai = time.perf_counter()
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_eff,
                min_dist=min_dist,
                random_state=42,
            )
            X_umap = reducer.fit_transform(X_scaled)
            selesai = time.perf_counter()

            hasil.append({
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "X_umap": X_umap.tolist(),
                "waktu_detik": round(selesai - mulai, 2),
                "error": None,
            })

    return hasil


# -----------------------------------------------------------
# Ekspor flag ketersediaan UMAP
# -----------------------------------------------------------
def is_umap_available():
    """Mengembalikan True jika pustaka umap-learn berhasil diimpor."""
    return UMAP_TERSEDIA
