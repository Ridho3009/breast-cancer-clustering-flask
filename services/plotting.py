"""
services/plotting.py
====================
Modul untuk membuat semua plot dan menyimpannya sebagai file PNG.
matplotlib menggunakan backend 'Agg' agar tidak membuka jendela GUI.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Wajib: backend tanpa GUI
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# Folder penyimpanan plot
PLOT_FOLDER = os.path.join("static", "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Warna latar plot
BG_FIG  = "#3d4b66"   # Latar figure
BG_AXES = "#3d4b66"   # Latar axes
CLR_GRID  = "#6b7c9e" # Garis grid
CLR_SPINE = "#5a6b8c" # Spine axes
CLR_TEXT  = "white"   # Semua teks, tick, label

# Palet warna klaster cerah agar jelas di atas latar biru-abu
PALET = ["#4c9aff", "#ff9f43", "#2ee6a6", "#ff6b9d",
         "#ffd166", "#a29bfe", "#fd79a8", "#55efc4"]

WARNA_NOISE = "#cbd5e1"  # Abu-abu muda untuk noise DBSCAN


def _get_colors(n):
    """Mengembalikan daftar warna cerah untuk n klaster."""
    return [PALET[i % len(PALET)] for i in range(n)]


def _simpan_plot(fig, nama_file):
    """Simpan figure ke folder static/plots/ dan tutup figure."""
    path = os.path.join(PLOT_FOLDER, nama_file)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"plots/{nama_file}"


# -----------------------------------------------------------
# Plot: Perbandingan distribusi fitur sebelum & sesudah scaling
# -----------------------------------------------------------
def plot_scaling_comparison(df_features, X_scaled, feature_names, nama_file="scaling_comparison.png"):
    """
    Membuat plot box yang membandingkan rentang nilai fitur
    sebelum dan sesudah StandardScaler diterapkan.
    Hanya menampilkan 10 fitur pertama agar tidak terlalu padat.
    """
    # Batasi ke 10 fitur pertama
    n_tampil = min(10, len(feature_names))
    fitur_tampil = feature_names[:n_tampil]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_FIG)
    fig.suptitle("Perbandingan Fitur: Sebelum vs Sesudah Scaling",
                 color=CLR_TEXT, fontsize=13, fontweight="bold", y=1.02)

    for ax in axes:
        ax.set_facecolor(BG_AXES)
        ax.tick_params(colors=CLR_TEXT, labelsize=7)
        ax.spines["bottom"].set_color(CLR_SPINE)
        ax.spines["left"].set_color(CLR_SPINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Sebelum scaling ---
    data_sebelum = [df_features[f].values for f in fitur_tampil]
    bp1 = axes[0].boxplot(
        data_sebelum,
        patch_artist=True,
        medianprops={"color": "#ff6b9d", "linewidth": 2},
        boxprops={"facecolor": "#51638a", "edgecolor": "#4c9aff"},
        whiskerprops={"color": "#4c9aff"},
        capprops={"color": "#4c9aff"},
        flierprops={"marker": ".", "color": "#ff9f43", "markersize": 3},
    )
    axes[0].set_title("Sebelum Scaling", color=CLR_TEXT, fontsize=11)
    axes[0].set_xticklabels(
        [f[:12] for f in fitur_tampil], rotation=45, ha="right", color=CLR_TEXT
    )
    axes[0].set_ylabel("Nilai", color=CLR_TEXT)

    # --- Sesudah scaling ---
    data_sesudah = [X_scaled[:, i] for i in range(n_tampil)]
    bp2 = axes[1].boxplot(
        data_sesudah,
        patch_artist=True,
        medianprops={"color": "#ff6b9d", "linewidth": 2},
        boxprops={"facecolor": "#51638a", "edgecolor": "#2ee6a6"},
        whiskerprops={"color": "#2ee6a6"},
        capprops={"color": "#2ee6a6"},
        flierprops={"marker": ".", "color": "#ff9f43", "markersize": 3},
    )
    axes[1].set_title("Sesudah Scaling (StandardScaler)", color=CLR_TEXT, fontsize=11)
    axes[1].set_xticklabels(
        [f[:12] for f in fitur_tampil], rotation=45, ha="right", color=CLR_TEXT
    )
    axes[1].set_ylabel("Nilai (z-score)", color=CLR_TEXT)

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: Elbow Curve (Inertia)
# -----------------------------------------------------------
def plot_elbow(kmeans_results, nama_file="kmeans_elbow.png"):
    """Membuat plot Elbow: sumbu X = K, sumbu Y = Inertia (WCSS)."""
    k_vals = [r["k"] for r in kmeans_results]
    inertias = [r["inertia"] for r in kmeans_results]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_FIG)
    ax.set_facecolor(BG_AXES)

    ax.plot(k_vals, inertias, "o-", color="#4c9aff", linewidth=2.5,
            markersize=8, markerfacecolor="#ff9f43", markeredgecolor="white")
    ax.set_xlabel("Jumlah Klaster (K)", color=CLR_TEXT, fontsize=11)
    ax.set_ylabel("Inertia (WCSS)", color=CLR_TEXT, fontsize=11)
    ax.set_title("Elbow Method – Mencari K Optimal", color=CLR_TEXT,
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors=CLR_TEXT)
    ax.set_xticks(k_vals)
    for sp in ax.spines.values():
        sp.set_color(CLR_SPINE)

    ax.grid(True, alpha=0.35, color=CLR_GRID)
    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: Silhouette Score
# -----------------------------------------------------------
def plot_silhouette(kmeans_results, k_terbaik, nama_file="kmeans_silhouette.png"):
    """Membuat plot Silhouette Score: sumbu X = K, sumbu Y = Silhouette."""
    k_vals = [r["k"] for r in kmeans_results]
    sil_vals = [r["silhouette"] for r in kmeans_results]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_FIG)
    ax.set_facecolor(BG_AXES)

    # Bar: kuning cerah untuk K terbaik, biru untuk lainnya
    colors = ["#ff9f43" if k == k_terbaik else "#4c9aff" for k in k_vals]
    bars = ax.bar(k_vals, sil_vals, color=colors, edgecolor=BG_FIG, linewidth=0.5)

    ax.set_xlabel("Jumlah Klaster (K)", color=CLR_TEXT, fontsize=11)
    ax.set_ylabel("Silhouette Score", color=CLR_TEXT, fontsize=11)
    ax.set_title("Silhouette Score per Nilai K", color=CLR_TEXT,
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors=CLR_TEXT)
    ax.set_xticks(k_vals)
    for sp in ax.spines.values():
        sp.set_color(CLR_SPINE)

    ax.grid(True, axis="y", alpha=0.35, color=CLR_GRID)
    ax.axhline(y=max(sil_vals), color="#ffd166", linewidth=1.2,
               linestyle="--", alpha=0.9, label=f"K terbaik = {k_terbaik}")
    ax.legend(facecolor=BG_AXES, labelcolor=CLR_TEXT)
    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: K-Distance Graph (untuk DBSCAN)
# -----------------------------------------------------------
def plot_k_distance(distances_sorted, eps_knee, min_samples,
                    nama_file="dbscan_kdistance.png"):
    """
    Membuat k-distance graph untuk membantu memilih nilai eps pada DBSCAN.
    Garis merah menandai estimasi posisi 'lutut' (knee point).
    """
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_FIG)
    ax.set_facecolor(BG_AXES)

    n = len(distances_sorted)
    ax.plot(range(n), distances_sorted, color="#4c9aff", linewidth=1.8)

    # Tandai posisi eps estimasi
    ax.axhline(y=eps_knee, color="#ff6b9d", linewidth=2, linestyle="--",
               label=f"Estimasi eps ≈ {eps_knee:.2f}")

    ax.set_xlabel("Titik Data (diurutkan)", color=CLR_TEXT, fontsize=11)
    ax.set_ylabel(f"Jarak ke Tetangga ke-{min_samples-1}", color=CLR_TEXT, fontsize=11)
    ax.set_title(f"K-Distance Graph (k={min_samples-1}) – Pilih Eps di 'Lutut'",
                 color=CLR_TEXT, fontsize=13, fontweight="bold")
    ax.tick_params(colors=CLR_TEXT)
    ax.legend(facecolor=BG_AXES, labelcolor=CLR_TEXT)
    ax.grid(True, alpha=0.35, color=CLR_GRID)
    for sp in ax.spines.values():
        sp.set_color(CLR_SPINE)

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: Scatter klaster DBSCAN
# -----------------------------------------------------------
def plot_dbscan_scatter(X_2d, dbscan_labels, eps, min_samples,
                        nama_file="dbscan_scatter.png"):
    """
    Scatter plot hasil DBSCAN dalam ruang 2D (menggunakan 2 komponen PCA).
    Titik noise (label -1) ditampilkan berwarna abu-abu.
    """
    labels_arr = np.array(dbscan_labels)
    unique_labels = sorted(set(labels_arr))
    n_klaster = len([l for l in unique_labels if l != -1])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_FIG)
    ax.set_facecolor(BG_AXES)

    colors = _get_colors(max(n_klaster, 1))
    color_idx = 0

    for label in unique_labels:
        mask = labels_arr == label
        if label == -1:
            # Noise
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=WARNA_NOISE, s=20, alpha=0.5,
                       label=f"Noise ({mask.sum()})", zorder=2)
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[color_idx % len(colors)]], s=25, alpha=0.8,
                       label=f"Klaster {label} ({mask.sum()})", zorder=3)
            color_idx += 1

    ax.set_title(f"DBSCAN (eps={eps}, min_samples={min_samples})\n"
                 f"{n_klaster} klaster, {int(np.sum(labels_arr == -1))} noise",
                 color=CLR_TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC 1", color=CLR_TEXT)
    ax.set_ylabel("PC 2", color=CLR_TEXT)
    ax.tick_params(colors=CLR_TEXT)
    ax.legend(facecolor=BG_AXES, labelcolor=CLR_TEXT, fontsize=8,
              loc="upper right")
    for sp in ax.spines.values():
        sp.set_color(CLR_SPINE)

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: PCA Explained Variance
# -----------------------------------------------------------
def plot_pca_variance(explained_ratio, cumulative, n_comp_85,
                      nama_file="pca_variance.png"):
    """
    Membuat plot gabungan:
    - Bar chart: Explained variance ratio per komponen
    - Garis: Cumulative variance dengan garis acuan 85%
    """
    n = len(explained_ratio)
    komponen = list(range(1, n + 1))

    fig, ax1 = plt.subplots(figsize=(11, 5), facecolor=BG_FIG)
    ax1.set_facecolor(BG_AXES)

    # Bar: variance per komponen
    ax1.bar(komponen, [v * 100 for v in explained_ratio],
            color="#4c9aff", alpha=0.8, label="Variance per Komponen (%)")

    ax1.set_xlabel("Nomor Komponen", color=CLR_TEXT, fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", color=CLR_TEXT, fontsize=11)
    ax1.set_title("PCA – Explained Variance Ratio", color=CLR_TEXT,
                  fontsize=13, fontweight="bold")
    ax1.tick_params(colors=CLR_TEXT)

    # Sumbu kedua untuk garis kumulatif
    ax2 = ax1.twinx()
    ax2.set_facecolor(BG_AXES)
    ax2.plot(komponen, [v * 100 for v in cumulative], "o-",
             color="#ff9f43", linewidth=2, markersize=4, label="Kumulatif (%)")
    ax2.axhline(y=85, color="#ffd166", linewidth=1.5, linestyle="--",
                label="Acuan 85%")
    ax2.axvline(x=n_comp_85, color="#ffd166", linewidth=1.2, linestyle=":",
                alpha=0.7)
    ax2.set_ylabel("Cumulative Variance (%)", color=CLR_TEXT, fontsize=11)
    ax2.tick_params(colors=CLR_TEXT)
    ax2.set_ylim(0, 105)

    # Anotasi jumlah komponen untuk 85%
    ax2.annotate(
        f"  {n_comp_85} komponen\n  untuk ≥85%",
        xy=(n_comp_85, 85),
        color="#ffd166",
        fontsize=9,
        va="bottom",
    )

    # Gabungkan legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=BG_AXES, labelcolor=CLR_TEXT, fontsize=9)

    for sp in ax1.spines.values():
        sp.set_color(CLR_SPINE)
    for sp in ax2.spines.values():
        sp.set_color(CLR_SPINE)

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: t-SNE scatter (3 perplexity berdampingan)
# -----------------------------------------------------------
def plot_tsne_grid(tsne_results, kmeans_labels, nama_file="tsne_grid.png"):
    """
    Menampilkan scatter plot t-SNE untuk 3 nilai perplexity secara berdampingan,
    diwarnai oleh label K-Means.
    """
    valid_results = [r for r in tsne_results if not r["dilewati"]]
    n_plot = len(valid_results)

    if n_plot == 0:
        # Tidak ada hasil yang valid
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=BG_FIG)
        ax.set_facecolor(BG_AXES)
        ax.text(0.5, 0.5, "Tidak ada hasil t-SNE yang dapat ditampilkan.",
                ha="center", va="center", color=CLR_TEXT)
        return _simpan_plot(fig, nama_file)

    fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 5),
                             facecolor=BG_FIG)
    if n_plot == 1:
        axes = [axes]

    fig.suptitle("t-SNE – Perbandingan Nilai Perplexity",
                 color=CLR_TEXT, fontsize=13, fontweight="bold")

    labels_arr = np.array(kmeans_labels)
    n_klaster = len(set(labels_arr))
    colors = _get_colors(n_klaster)

    for ax, result in zip(axes, valid_results):
        ax.set_facecolor(BG_AXES)
        X = np.array(result["X_tsne"])

        for i in range(n_klaster):
            mask = labels_arr == i
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[colors[i]], s=20, alpha=0.8,
                       label=f"K-{i}")

        ax.set_title(
            f"Perplexity = {result['perplexity']}\n"
            f"Waktu: {result['waktu_detik']:.1f} detik",
            color=CLR_TEXT, fontsize=10
        )
        ax.tick_params(colors=CLR_TEXT)
        ax.set_xlabel("Dim 1", color=CLR_TEXT)
        ax.set_ylabel("Dim 2", color=CLR_TEXT)
        for sp in ax.spines.values():
            sp.set_color(CLR_SPINE)
        ax.grid(True, alpha=0.2, color=CLR_GRID)

    axes[0].legend(facecolor=BG_AXES, labelcolor=CLR_TEXT, fontsize=7)
    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: UMAP grid (n_neighbors x min_dist)
# -----------------------------------------------------------
def plot_umap_grid(umap_results, kmeans_labels, n_neighbors_list, min_dist_list,
                   nama_file="umap_grid.png"):
    """
    Menampilkan grid scatter plot UMAP: baris = n_neighbors, kolom = min_dist.
    """
    n_rows = len(n_neighbors_list)
    n_cols = len(min_dist_list)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             facecolor=BG_FIG)
    fig.suptitle("UMAP – Grid n_neighbors × min_dist",
                 color=CLR_TEXT, fontsize=13, fontweight="bold")

    labels_arr = np.array(kmeans_labels)
    n_klaster = len(set(labels_arr))
    colors = _get_colors(n_klaster)

    # Pastikan axes selalu 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    result_idx = 0
    for i, nn in enumerate(n_neighbors_list):
        for j, md in enumerate(min_dist_list):
            ax = axes[i][j]
            ax.set_facecolor(BG_AXES)

            if result_idx < len(umap_results):
                res = umap_results[result_idx]
                X = np.array(res["X_umap"])

                for k in range(n_klaster):
                    mask = labels_arr == k
                    ax.scatter(X[mask, 0], X[mask, 1],
                               c=[colors[k]], s=15, alpha=0.8)

                ax.set_title(
                    f"n_neighbors={nn}, min_dist={md}\n"
                    f"Waktu: {res['waktu_detik']:.1f}s",
                    color=CLR_TEXT, fontsize=9
                )

            ax.tick_params(colors=CLR_TEXT, labelsize=7)
            ax.set_xlabel("Dim 1", color=CLR_TEXT, fontsize=8)
            ax.set_ylabel("Dim 2", color=CLR_TEXT, fontsize=8)
            ax.grid(True, alpha=0.2, color=CLR_GRID)
            for sp in ax.spines.values():
                sp.set_color(CLR_SPINE)

            result_idx += 1

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Plot: Visualisasi Gabungan (Combined)
# -----------------------------------------------------------
def plot_combined(
    X_pca, X_tsne_30, X_umap_15,
    kmeans_labels, dbscan_labels,
    original_labels=None,
    nama_file="combined_visualization.png"
):
    """
    Subplot 2 (atau 3) baris × 3 kolom:
    - Kolom: PCA | t-SNE (perp=30) | UMAP (nn=15, md=0.1)
    - Baris 1: diwarnai label K-Means
    - Baris 2: diwarnai label DBSCAN (noise=abu-abu)
    - Baris 3 (jika ada): label asli
    """
    ada_original = original_labels is not None
    n_baris = 3 if ada_original else 2
    n_kolom = 3

    judul_baris = ["K-Means", "DBSCAN", "Label Asli"]
    judul_kolom = ["PCA", "t-SNE (perp=30)", "UMAP (nn=15, md=0.1)"]

    fig, axes = plt.subplots(n_baris, n_kolom,
                             figsize=(5 * n_kolom, 4.5 * n_baris),
                             facecolor=BG_FIG)
    fig.suptitle("Visualisasi Gabungan – Semua Metode Reduksi & Klasterisasi",
                 color=CLR_TEXT, fontsize=14, fontweight="bold", y=1.01)

    # Data reduksi dimensi (bisa None jika belum tersedia)
    datasets = [X_pca, X_tsne_30, X_umap_15]

    # Siapkan label per baris
    km_arr = np.array(kmeans_labels)
    n_km = len(set(km_arr))
    km_colors = _get_colors(n_km)

    db_arr = np.array(dbscan_labels)
    db_unique = sorted(set(db_arr) - {-1})
    n_db = len(db_unique)
    db_colors = _get_colors(max(n_db, 1))

    if ada_original:
        orig_arr = np.array(original_labels)
        orig_unique = sorted(set(orig_arr))
        n_orig = len(orig_unique)
        orig_colors = _get_colors(n_orig)

    for baris in range(n_baris):
        for kolom in range(n_kolom):
            ax = axes[baris][kolom]
            ax.set_facecolor("#16213e")
            X_2d = datasets[kolom]

            if X_2d is None:
                ax.text(0.5, 0.5, "Data tidak\ntersedia",
                        ha="center", va="center", color=CLR_TEXT, fontsize=9)
                ax.set_title(f"{judul_baris[baris]}\n{judul_kolom[kolom]}",
                             color=CLR_TEXT, fontsize=9)
                ax.set_facecolor(BG_AXES)
                ax.tick_params(colors=CLR_TEXT, labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color(CLR_SPINE)
                continue

            X_arr = np.array(X_2d)

            # --- Baris 0: K-Means ---
            if baris == 0:
                for i in range(n_km):
                    mask = km_arr == i
                    ax.scatter(X_arr[mask, 0], X_arr[mask, 1],
                               c=[km_colors[i]], s=15, alpha=0.8,
                               label=f"K-{i}")

            # --- Baris 1: DBSCAN ---
            elif baris == 1:
                # Noise dulu (agar tertimpa klaster)
                noise_mask = db_arr == -1
                if noise_mask.any():
                    ax.scatter(X_arr[noise_mask, 0], X_arr[noise_mask, 1],
                               c=WARNA_NOISE, s=10, alpha=0.4, label="Noise")
                for ci, label in enumerate(db_unique):
                    mask = db_arr == label
                    ax.scatter(X_arr[mask, 0], X_arr[mask, 1],
                               c=[db_colors[ci % len(db_colors)]], s=15, alpha=0.8,
                               label=f"Klaster {label}")

            # --- Baris 2: Label Asli ---
            elif baris == 2 and ada_original:
                for ci, label in enumerate(orig_unique):
                    mask = orig_arr == label
                    ax.scatter(X_arr[mask, 0], X_arr[mask, 1],
                               c=[orig_colors[ci]], s=15, alpha=0.8,
                               label=str(label))

            ax.set_facecolor(BG_AXES)
            ax.set_title(f"{judul_baris[baris]} — {judul_kolom[kolom]}",
                         color=CLR_TEXT, fontsize=9)
            ax.tick_params(colors=CLR_TEXT, labelsize=7)
            ax.set_xlabel("Dim 1", color=CLR_TEXT, fontsize=8)
            ax.set_ylabel("Dim 2", color=CLR_TEXT, fontsize=8)
            ax.grid(True, alpha=0.2, color=CLR_GRID)
            for sp in ax.spines.values():
                sp.set_color(CLR_SPINE)

            if kolom == 0:
                ax.legend(facecolor=BG_AXES, labelcolor=CLR_TEXT,
                          fontsize=7, markerscale=1.5)

    plt.tight_layout()
    return _simpan_plot(fig, nama_file)


# -----------------------------------------------------------
# Hapus semua plot lama
# -----------------------------------------------------------
def clear_all_plots():
    """Menghapus semua file PNG di folder static/plots/."""
    for fname in os.listdir(PLOT_FOLDER):
        if fname.endswith(".png"):
            try:
                os.remove(os.path.join(PLOT_FOLDER, fname))
            except OSError:
                pass
