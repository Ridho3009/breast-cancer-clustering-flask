"""
app.py
======
File utama aplikasi Flask.
Jalankan dengan: python app.py
"""

import os
import json
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.utils import secure_filename

# Import modul layanan (services)
from services.data import (
    load_default_dataset,
    load_uploaded_csv,
    save_active_data,
    load_active_data,
    delete_old_data,
    scale_features,
    analyze_data_quality,
)
from services.clustering import (
    run_kmeans_analysis,
    run_dbscan,
    run_dbscan_combinations,
    compute_k_distance,
    estimate_eps_from_kdistance,
)
from services.reduction import run_pca, run_tsne, run_umap, is_umap_available
from services.plotting import (
    plot_scaling_comparison,
    plot_elbow,
    plot_silhouette,
    plot_k_distance,
    plot_dbscan_scatter,
    plot_pca_variance,
    plot_tsne_grid,
    plot_umap_grid,
    plot_combined,
    clear_all_plots,
)

# -----------------------------------------------------------
# Inisialisasi Aplikasi Flask
# -----------------------------------------------------------
app = Flask(__name__)
# Kunci sesi: gunakan variabel lingkungan jika ada (lebih aman di produksi)
app.secret_key = os.environ.get("SECRET_KEY", "kunci-lokal-tugas")

# Batas ukuran unggahan: 5 MB (ditegakkan Flask sebelum file diproses)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Folder untuk file unggahan
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join("static", "plots"), exist_ok=True)

# -----------------------------------------------------------
# Cache sederhana di memori (agar t-SNE & UMAP tidak dihitung ulang)
# -----------------------------------------------------------
_cache = {}


def get_cache(key):
    """Ambil nilai dari cache."""
    return _cache.get(key)


def set_cache(key, value):
    """Simpan nilai ke cache."""
    _cache[key] = value


def clear_cache():
    """Hapus semua cache saat sumber data diganti."""
    _cache.clear()


# -----------------------------------------------------------
# Fungsi bantu: ambil data aktif dari sesi
# -----------------------------------------------------------
def get_active_data():
    """
    Mengambil data aktif berdasarkan data_id yang tersimpan di sesi.
    Jika belum ada data atau file sesi tidak ditemukan, otomatis muat dataset default.
    """
    data_id = session.get("data_id")
    if data_id:
        df_feat, df_lab, meta = load_active_data(data_id, UPLOAD_FOLDER)
        if df_feat is not None:
            return df_feat, df_lab, meta

    # Otomatis muat dataset bawaan Breast Cancer Wisconsin
    df_features, df_labels, feature_names = load_default_dataset()
    data_id = save_active_data(df_features, df_labels, feature_names, UPLOAD_FOLDER)
    session["data_id"] = data_id
    session["sumber_data"] = "default"
    metadata = {
        "feature_names": feature_names,
        "has_labels": True,
        "label_name": "target",
        "n_samples": len(df_features),
        "n_features": len(feature_names),
    }
    return df_features, df_labels, metadata


def get_scaled_data():
    """
    Mengambil data yang sudah di-scaling dari cache atau menghitungnya.
    Returns (X_scaled, df_features, df_labels, metadata) atau (None, ...).
    """
    cached = get_cache("scaled_data")
    if cached is not None:
        return cached

    df_features, df_labels, metadata = get_active_data()
    if df_features is None:
        return None, None, None, None

    X_scaled, scaler, stats_before, stats_after = scale_features(df_features)
    result = (X_scaled, df_features, df_labels, metadata)
    set_cache("scaled_data", result)
    return result


# -----------------------------------------------------------
# HALAMAN 1: Beranda
# -----------------------------------------------------------
@app.route("/")
def beranda():
    """Halaman utama: judul, deskripsi, dan ringkasan dataset."""
    df_features, df_labels, metadata = get_active_data()

    info_dataset = None
    if df_features is not None and metadata is not None:
        info_dataset = {
            "n_sampel": metadata["n_samples"],
            "n_fitur": metadata["n_features"],
            "ada_label": metadata["has_labels"],
            "nama_label": metadata.get("label_name"),
            "sumber": session.get("sumber_data", "default"),
        }

    return render_template("beranda.html", info_dataset=info_dataset)


# -----------------------------------------------------------
# HALAMAN 2: Preprocessing
# -----------------------------------------------------------
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    """
    Halaman preprocessing: pilih sumber data, tampilkan info kualitas,
    lakukan scaling, dan tampilkan plot perbandingan.
    """
    if request.method == "POST":
        sumber = request.form.get("sumber_data", "default")
        label_column = request.form.get("label_column", "").strip() or None

        # Hapus data lama dan cache
        data_id_lama = session.get("data_id")
        delete_old_data(data_id_lama, UPLOAD_FOLDER)
        clear_cache()
        clear_all_plots()

        if sumber == "default":
            # Muat dataset bawaan Breast Cancer
            df_features, df_labels, feature_names = load_default_dataset()
            warnings = []

        elif sumber == "upload":
            # Muat file CSV yang diunggah
            file = request.files.get("csv_file")
            hasil = load_uploaded_csv(file, label_column=label_column)

            if hasil["error"]:
                flash(f"Error: {hasil['error']}", "danger")
                return redirect(url_for("preprocessing"))

            df_features = hasil["df_features"]
            df_labels = hasil["df_labels"]
            feature_names = hasil["feature_names"]
            warnings = hasil["warnings"]

            for w in warnings:
                flash(w, "warning")
        else:
            flash("Sumber data tidak valid.", "danger")
            return redirect(url_for("preprocessing"))

        # Simpan data aktif ke file sementara
        data_id = save_active_data(df_features, df_labels, feature_names, UPLOAD_FOLDER)
        session["data_id"] = data_id
        session["sumber_data"] = sumber

        flash("Data berhasil dimuat!", "success")
        return redirect(url_for("preprocessing"))

    # --- GET: tampilkan halaman ---
    df_features, df_labels, metadata = get_active_data()

    if df_features is None:
        # Muat dataset default jika belum ada data
        df_features, df_labels, feature_names = load_default_dataset()
        data_id = save_active_data(df_features, df_labels, feature_names, UPLOAD_FOLDER)
        session["data_id"] = data_id
        session["sumber_data"] = "default"
        metadata = {
            "feature_names": feature_names,
            "has_labels": True,
            "label_name": "target",
            "n_samples": len(df_features),
            "n_features": len(feature_names),
        }

    # Analisis kualitas data
    kualitas = analyze_data_quality(df_features)

    # Scaling
    X_scaled, scaler, stats_before, stats_after = scale_features(df_features)
    set_cache("scaled_data", (X_scaled, df_features, df_labels, metadata))

    # Buat plot scaling
    feature_names = metadata["feature_names"]
    plot_path = plot_scaling_comparison(df_features, X_scaled, feature_names)

    # Preview 5 baris pertama
    preview_html = df_features.head(5).to_html(
        classes="table table-sm table-striped table-dark",
        border=0, index=True
    )

    # Siapkan tabel statistik
    stats_tabel = []
    for fitur in feature_names[:15]:  # Tampilkan max 15 fitur
        stats_tabel.append({
            "fitur": fitur,
            "mean_sebelum": stats_before["mean"].get(fitur, "-"),
            "std_sebelum": stats_before["std"].get(fitur, "-"),
            "mean_sesudah": stats_after["mean"].get(fitur, "-"),
            "std_sesudah": stats_after["std"].get(fitur, "-"),
        })

    return render_template(
        "preprocessing.html",
        kualitas=kualitas,
        preview_html=preview_html,
        stats_tabel=stats_tabel,
        plot_path=plot_path,
        metadata=metadata,
        sumber_aktif=session.get("sumber_data", "default"),
    )


# -----------------------------------------------------------
# HALAMAN 3: K-Means
# -----------------------------------------------------------
@app.route("/kmeans")
def kmeans():
    """Halaman K-Means: analisis untuk K=2 sampai 10."""
    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Cek cache agar tidak menghitung ulang
    cached_km = get_cache("kmeans_results")
    if cached_km is None:
        kmeans_results, k_terbaik = run_kmeans_analysis(X_scaled)
        set_cache("kmeans_results", (kmeans_results, k_terbaik))
    else:
        kmeans_results, k_terbaik = cached_km

    # Buat plot
    path_elbow = plot_elbow(kmeans_results)
    path_sil = plot_silhouette(kmeans_results, k_terbaik)

    # Label K-Means terbaik untuk dipakai di halaman lain
    km_labels_terbaik = next(
        r["labels"] for r in kmeans_results if r["k"] == k_terbaik
    )
    set_cache("km_labels_terbaik", km_labels_terbaik)
    set_cache("k_terbaik", k_terbaik)

    return render_template(
        "kmeans.html",
        kmeans_results=kmeans_results,
        k_terbaik=k_terbaik,
        path_elbow=path_elbow,
        path_sil=path_sil,
    )


# -----------------------------------------------------------
# HALAMAN 4: DBSCAN
# -----------------------------------------------------------
@app.route("/dbscan", methods=["GET", "POST"])
def dbscan():
    """Halaman DBSCAN: k-distance graph, tabel kombinasi, form interaktif."""
    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Pastikan ada label K-Means untuk visualisasi 2D via PCA
    pca_result = get_cache("pca_result")
    if pca_result is None:
        pca_result = run_pca(X_scaled)
        set_cache("pca_result", pca_result)

    X_pca_2d = np.array(pca_result["X_pca"])

    # Hitung k-distance graph (min_samples default = 5)
    min_samples_default = 5
    k_for_distance = min_samples_default - 1

    cached_kdist = get_cache("k_distance")
    if cached_kdist is None:
        distances_sorted = compute_k_distance(X_scaled, k=k_for_distance)
        set_cache("k_distance", distances_sorted)
    else:
        distances_sorted = cached_kdist

    # Estimasi eps
    eps_knee, eps_lower, eps_upper = estimate_eps_from_kdistance(distances_sorted)

    # Plot k-distance
    path_kdist = plot_k_distance(distances_sorted, eps_knee, min_samples_default)

    # Jalankan 3 kombinasi default
    cached_combinations = get_cache("dbscan_combinations")
    if cached_combinations is None:
        eps_list = [eps_lower, eps_knee, eps_upper]
        ms_list = [min_samples_default] * 3
        kombinasi = run_dbscan_combinations(X_scaled, eps_list, ms_list)
        set_cache("dbscan_combinations", kombinasi)
    else:
        kombinasi = cached_combinations

    # Parameter default untuk visualisasi gabungan (gunakan eps_knee)
    default_dbscan = kombinasi[1]  # Kombinasi tengah (eps_knee)
    set_cache("dbscan_default", default_dbscan)

    # Scatter plot DBSCAN default
    path_scatter = plot_dbscan_scatter(
        X_pca_2d,
        default_dbscan["labels"],
        default_dbscan["eps"],
        default_dbscan["min_samples"]
    )

    # --- POST: pengguna mencoba parameter sendiri ---
    custom_result = None
    if request.method == "POST":
        try:
            eps_custom = float(request.form.get("eps", eps_knee))
            ms_custom = int(request.form.get("min_samples", min_samples_default))
            custom_result = run_dbscan(X_scaled, eps=eps_custom, min_samples=ms_custom)

            if custom_result["n_clusters"] == 0:
                flash(
                    f"Dengan eps={eps_custom} dan min_samples={ms_custom}, "
                    "DBSCAN tidak menemukan klaster (semua titik dianggap noise). "
                    "Coba nilai eps yang lebih besar atau min_samples yang lebih kecil.",
                    "warning"
                )

            # Plot scatter untuk parameter custom
            path_scatter_custom = plot_dbscan_scatter(
                X_pca_2d,
                custom_result["labels"],
                custom_result["eps"],
                custom_result["min_samples"],
                nama_file="dbscan_scatter_custom.png"
            )
            custom_result["plot_path"] = path_scatter_custom

        except (ValueError, TypeError) as e:
            flash(f"Parameter tidak valid: {str(e)}", "danger")

    return render_template(
        "dbscan.html",
        eps_knee=eps_knee,
        eps_lower=eps_lower,
        eps_upper=eps_upper,
        min_samples_default=min_samples_default,
        kombinasi=kombinasi,
        path_kdist=path_kdist,
        path_scatter=path_scatter,
        custom_result=custom_result,
        default_dbscan=default_dbscan,
    )


# -----------------------------------------------------------
# HALAMAN 5: PCA
# -----------------------------------------------------------
@app.route("/pca")
def pca():
    """Halaman PCA: explained variance dan komponen minimum untuk 85%."""
    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Cek cache
    pca_result = get_cache("pca_result")
    if pca_result is None:
        pca_result = run_pca(X_scaled)
        set_cache("pca_result", pca_result)

    # Plot explained variance
    path_variance = plot_pca_variance(
        pca_result["explained_ratio"],
        pca_result["cumulative"],
        pca_result["n_comp_85"]
    )

    # Siapkan tabel variance per komponen (max 15)
    tabel_variance = []
    for i, (ev, cum) in enumerate(zip(
        pca_result["explained_ratio"][:15],
        pca_result["cumulative"][:15]
    )):
        tabel_variance.append({
            "komponen": i + 1,
            "variance_pct": round(ev * 100, 2),
            "kumulatif_pct": round(cum * 100, 2),
        })

    return render_template(
        "pca.html",
        pca_result=pca_result,
        path_variance=path_variance,
        tabel_variance=tabel_variance,
    )


# -----------------------------------------------------------
# HALAMAN 6: t-SNE
# -----------------------------------------------------------
@app.route("/tsne")
def tsne():
    """Halaman t-SNE: perbandingan 3 nilai perplexity."""
    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Pastikan label K-Means tersedia
    km_labels = get_cache("km_labels_terbaik")
    if km_labels is None:
        kmeans_results, k_terbaik = run_kmeans_analysis(X_scaled)
        km_labels = next(r["labels"] for r in kmeans_results if r["k"] == k_terbaik)
        set_cache("kmeans_results", (kmeans_results, k_terbaik))
        set_cache("km_labels_terbaik", km_labels)
        set_cache("k_terbaik", k_terbaik)

    # Cek cache t-SNE — cache dianggap valid hanya jika memuat ketiga perplexity.
    # Ini mencegah cache yang diisi oleh /visualisasi (hanya perp=30) menimpa.
    PERPLEXITIES = [5, 30, 50]
    tsne_results = get_cache("tsne_results")
    cache_perp_lengkap = (
        tsne_results is not None
        and {r["perplexity"] for r in tsne_results} >= set(PERPLEXITIES)
    )
    if not cache_perp_lengkap:
        tsne_results = run_tsne(X_scaled, perplexities=PERPLEXITIES)
        set_cache("tsne_results", tsne_results)

    # Plot
    path_tsne = plot_tsne_grid(tsne_results, km_labels)

    # Simpan hasil t-SNE perplexity=30 untuk visualisasi gabungan
    X_tsne_30 = None
    for r in tsne_results:
        if r["perplexity"] == 30 and not r["dilewati"]:
            X_tsne_30 = r["X_tsne"]
            break
    set_cache("X_tsne_30", X_tsne_30)

    return render_template(
        "tsne.html",
        tsne_results=tsne_results,
        path_tsne=path_tsne,
        km_labels=km_labels,
    )


# -----------------------------------------------------------
# HALAMAN 7: UMAP
# -----------------------------------------------------------
@app.route("/umap")
def umap_page():
    """Halaman UMAP: grid n_neighbors × min_dist."""
    if not is_umap_available():
        flash(
            "Pustaka UMAP tidak ditemukan. Instal dengan: pip install umap-learn",
            "danger"
        )
        return render_template("umap.html", umap_tersedia=False,
                               umap_results=[], tsne_results=[])

    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Label K-Means untuk pewarnaan
    km_labels = get_cache("km_labels_terbaik")
    if km_labels is None:
        kmeans_results, k_terbaik = run_kmeans_analysis(X_scaled)
        km_labels = next(r["labels"] for r in kmeans_results if r["k"] == k_terbaik)
        set_cache("kmeans_results", (kmeans_results, k_terbaik))
        set_cache("km_labels_terbaik", km_labels)
        set_cache("k_terbaik", k_terbaik)

    # Parameter UMAP
    n_neighbors_list = [5, 15, 50]
    min_dist_list = [0.1, 0.5]
    # Semua kombinasi yang diharapkan
    KOMBINASI_UMAP = {
        (nn, md) for nn in n_neighbors_list for md in min_dist_list
    }

    # Cek cache UMAP — cache valid hanya jika semua 6 kombinasi tersedia.
    # Ini mencegah cache dari /visualisasi (hanya nn=15, md=0.1) menimpa.
    umap_results = get_cache("umap_results")
    kombinasi_cache = (
        {(r["n_neighbors"], r["min_dist"]) for r in umap_results}
        if isinstance(umap_results, list) else set()
    )
    if kombinasi_cache < KOMBINASI_UMAP:
        umap_results = run_umap(X_scaled, n_neighbors_list, min_dist_list)
        if isinstance(umap_results, dict) and "error" in umap_results:
            flash(umap_results["error"], "danger")
            return render_template("umap.html", umap_tersedia=False,
                                   umap_results=[], tsne_results=[])
        set_cache("umap_results", umap_results)

    # Plot UMAP grid
    path_umap = plot_umap_grid(umap_results, km_labels, n_neighbors_list, min_dist_list)

    # Simpan UMAP nn=15, md=0.1 untuk visualisasi gabungan
    X_umap_15 = None
    for res in umap_results:
        if res["n_neighbors"] == 15 and res["min_dist"] == 0.1:
            X_umap_15 = res["X_umap"]
            break
    set_cache("X_umap_15", X_umap_15)

    # Data t-SNE untuk tabel perbandingan waktu
    tsne_results = get_cache("tsne_results") or []

    return render_template(
        "umap.html",
        umap_tersedia=True,
        umap_results=umap_results,
        tsne_results=tsne_results,
        path_umap=path_umap,
        n_neighbors_list=n_neighbors_list,
        min_dist_list=min_dist_list,
    )


# -----------------------------------------------------------
# HALAMAN 8: Visualisasi Gabungan
# -----------------------------------------------------------
@app.route("/visualisasi")
def visualisasi():
    """Halaman visualisasi gabungan: subplot 2x3 atau 3x3."""
    X_scaled, df_features, df_labels, metadata = get_scaled_data()

    if X_scaled is None:
        flash("Silakan muat data terlebih dahulu di halaman Preprocessing.", "warning")
        return redirect(url_for("preprocessing"))

    # Pastikan semua hasil tersedia
    pca_result = get_cache("pca_result")
    if pca_result is None:
        pca_result = run_pca(X_scaled)
        set_cache("pca_result", pca_result)

    km_labels = get_cache("km_labels_terbaik")
    if km_labels is None:
        kmeans_results, k_terbaik = run_kmeans_analysis(X_scaled)
        km_labels = next(r["labels"] for r in kmeans_results if r["k"] == k_terbaik)
        set_cache("kmeans_results", (kmeans_results, k_terbaik))
        set_cache("km_labels_terbaik", km_labels)
        set_cache("k_terbaik", k_terbaik)

    dbscan_default = get_cache("dbscan_default")
    if dbscan_default is None:
        # Hitung DBSCAN dengan parameter default
        distances_sorted = compute_k_distance(X_scaled, k=4)
        eps_knee, _, _ = estimate_eps_from_kdistance(distances_sorted)
        dbscan_default = run_dbscan(X_scaled, eps=eps_knee, min_samples=5)
        set_cache("dbscan_default", dbscan_default)

    X_tsne_30 = get_cache("X_tsne_30")
    if X_tsne_30 is None:
        # Cek jika tsne_results sudah ada di cache
        cached_tsne = get_cache("tsne_results")
        if cached_tsne:
            for r in cached_tsne:
                if r.get("perplexity") == 30 and not r.get("dilewati"):
                    X_tsne_30 = r["X_tsne"]
                    set_cache("X_tsne_30", X_tsne_30)
                    break
        if X_tsne_30 is None:
            # Hitung hanya perplexity=30 untuk keperluan plot gabungan.
            # JANGAN simpan ke "tsne_results" agar tidak menimpa cache 3-perplexity
            # yang dihasilkan oleh halaman /tsne.
            _tsne_tmp = run_tsne(X_scaled, perplexities=[30])
            if _tsne_tmp and not _tsne_tmp[0]["dilewati"]:
                X_tsne_30 = _tsne_tmp[0]["X_tsne"]
                set_cache("X_tsne_30", X_tsne_30)

    X_umap_15 = get_cache("X_umap_15")
    if X_umap_15 is None and is_umap_available():
        # Cek jika umap_results sudah ada di cache
        cached_umap = get_cache("umap_results")
        if isinstance(cached_umap, list):
            for res in cached_umap:
                if res.get("n_neighbors") == 15 and res.get("min_dist") == 0.1:
                    X_umap_15 = res["X_umap"]
                    set_cache("X_umap_15", X_umap_15)
                    break
        if X_umap_15 is None:
            # Hitung hanya nn=15, md=0.1 untuk plot gabungan.
            # JANGAN simpan ke "umap_results" agar tidak menimpa cache 6-kombinasi
            # yang dihasilkan oleh halaman /umap.
            _umap_tmp = run_umap(X_scaled, n_neighbors_list=[15], min_dist_list=[0.1])
            if isinstance(_umap_tmp, list) and len(_umap_tmp) > 0:
                X_umap_15 = _umap_tmp[0]["X_umap"]
                set_cache("X_umap_15", X_umap_15)

    # Label asli (jika ada)
    original_labels = None
    if df_labels is not None:
        original_labels = df_labels.tolist()

    # Buat plot gabungan
    path_combined = plot_combined(
        X_pca=pca_result["X_pca"],
        X_tsne_30=X_tsne_30,
        X_umap_15=X_umap_15,
        kmeans_labels=km_labels,
        dbscan_labels=dbscan_default["labels"],
        original_labels=original_labels,
    )

    k_terbaik = get_cache("k_terbaik") or "?"
    dbscan_info = {
        "eps": dbscan_default["eps"],
        "min_samples": dbscan_default["min_samples"],
        "n_clusters": dbscan_default["n_clusters"],
        "n_noise": dbscan_default["n_noise"],
    }

    return render_template(
        "visualisasi.html",
        path_combined=path_combined,
        k_terbaik=k_terbaik,
        dbscan_info=dbscan_info,
        ada_label=original_labels is not None,
        umap_tersedia=is_umap_available(),
    )


# -----------------------------------------------------------
# Jalankan Aplikasi
# -----------------------------------------------------------
if __name__ == "__main__":
    # host=127.0.0.1: hanya bisa diakses dari mesin lokal (lebih aman)
    # debug=False  : wajib untuk kode yang diunggah ke GitHub
    app.run(host="127.0.0.1", port=5000, debug=False)
