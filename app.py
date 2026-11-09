
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
from datetime import datetime

from preprocessing import preprocessing


# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Blibli",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Analisis Sentimen Ulasan Blibli")
st.caption(
    "Analisis sentimen menggunakan TF-IDF "
    "dan Multinomial Naive Bayes"
)


# =====================================================
# MEMUAT MODEL DAN EVALUASI
# =====================================================

@st.cache_resource
def muat_model():
    return joblib.load("model_sentimen_blibli.joblib")


@st.cache_data
def muat_evaluasi():
    with open("evaluasi.json", "r", encoding="utf-8") as f:
        return json.load(f)


model = muat_model()
evaluasi = muat_evaluasi()


# =====================================================
# RIWAYAT ANALISIS
# =====================================================

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []


# =====================================================
# FUNGSI ANALISIS SENTIMEN
# =====================================================

def analisis_komentar(komentar):

    hasil = []

    for teks in komentar:

        bersih = preprocessing(teks)

        data = {
            "Waktu Analisis": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "Komentar": teks,
            "Teks Bersih": bersih,
            "Sentimen": "Teks tidak cukup informatif",
            "Probabilitas Tertinggi (%)": None,
            "Probabilitas Positif (%)": None,
            "Probabilitas Netral (%)": None,
            "Probabilitas Negatif (%)": None
        }

        # Teks kosong atau tidak memiliki kata informatif
        if not bersih.strip():
            hasil.append(data)
            continue

        probabilitas = model.predict_proba([bersih])[0]
        kelas = list(model.classes_)

        indeks = int(np.argmax(probabilitas))
        prediksi = kelas[indeks]

        data["Sentimen"] = prediksi
        data["Probabilitas Tertinggi (%)"] = round(
            float(probabilitas[indeks] * 100), 2
        )

        for i, k in enumerate(kelas):
            kolom = f"Probabilitas {k} (%)"
            data[kolom] = round(
                float(probabilitas[i] * 100), 2
            )

        hasil.append(data)

    return hasil


# =====================================================
# TIGA TAB UTAMA
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "💬 Analisis Komentar",
    "📋 Riwayat & Excel",
    "📊 Evaluasi Model"
])


# =====================================================
# TAB 1: ANALISIS KOMENTAR
# =====================================================

with tab1:

    st.subheader("Analisis Sentimen Komentar")

    st.write(
        "Masukkan satu atau beberapa komentar. "
        "Gunakan satu baris untuk setiap komentar."
    )

    teks_input = st.text_area(
        "Masukkan komentar:",
        height=180,
        placeholder=(
            "Aplikasi Blibli sangat mudah digunakan\n"
            "Pengiriman barang sangat lambat\n"
            "Aplikasi biasa saja"
        )
    )

    if st.button(
        "🔍 Analisis Sentimen",
        type="primary",
        use_container_width=True
    ):

        komentar = [
            baris.strip()
            for baris in teks_input.splitlines()
            if baris.strip()
        ]

        if not komentar:
            st.warning("Masukkan komentar terlebih dahulu.")

        else:
            hasil = analisis_komentar(komentar)

            st.session_state.riwayat.extend(hasil)

            jumlah_berhasil = sum(
                1 for h in hasil
                if h["Sentimen"] != "Teks tidak cukup informatif"
            )

            st.success(
                f"{jumlah_berhasil} dari {len(hasil)} "
                "komentar berhasil dianalisis."
            )

            st.dataframe(
                pd.DataFrame(hasil),
                use_container_width=True,
                hide_index=True
            )


# =====================================================
# TAB 2: RIWAYAT DAN EXCEL
# =====================================================

with tab2:

    st.subheader("Riwayat Hasil Analisis")

    if st.session_state.riwayat:

        riwayat = pd.DataFrame(
            st.session_state.riwayat
        )

        jumlah = len(riwayat)

        positif = (
            riwayat["Sentimen"] == "Positif"
        ).sum()

        netral = (
            riwayat["Sentimen"] == "Netral"
        ).sum()

        negatif = (
            riwayat["Sentimen"] == "Negatif"
        ).sum()

        tidak_informatif = (
            riwayat["Sentimen"] ==
            "Teks tidak cukup informatif"
        ).sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Komentar", jumlah)
        c2.metric("Positif", int(positif))
        c3.metric("Netral", int(netral))
        c4.metric("Negatif", int(negatif))

        st.write("### Tabel Riwayat Analisis")

        st.dataframe(
            riwayat,
            use_container_width=True,
            hide_index=True
        )

        # Membuat file Excel
        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            riwayat.to_excel(
                writer,
                sheet_name="Hasil Analisis",
                index=False
            )

        st.download_button(
            label="⬇️ Unduh Hasil ke Excel",
            data=output.getvalue(),
            file_name="hasil_analisis_sentimen_blibli.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

        if st.button(
            "🗑️ Hapus Riwayat Analisis",
            use_container_width=True
        ):

            st.session_state.riwayat = []
            st.rerun()

    else:

        st.info(
            "Belum ada riwayat analisis. "
            "Silakan masukkan komentar pada tab Analisis Komentar."
        )


# =====================================================
# TAB 3: EVALUASI MODEL
# =====================================================

with tab3:

    st.subheader("Evaluasi Model Multinomial Naive Bayes")

    # Nilai evaluasi split tunggal
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Akurasi",
        f"{evaluasi['akurasi'] * 100:.2f}%"
    )

    c2.metric(
        "Precision Macro",
        f"{evaluasi['macro_precision'] * 100:.2f}%"
    )

    c3.metric(
        "Recall Macro",
        f"{evaluasi['macro_recall'] * 100:.2f}%"
    )

    c4.metric(
        "F1-Score Macro",
        f"{evaluasi['macro_f1'] * 100:.2f}%"
    )

    st.write("### Evaluasi Nested 10-Fold Cross-Validation")

    n1, n2 = st.columns(2)

    n1.metric(
        "Rata-rata Akurasi",
        f"{evaluasi['nested_akurasi_mean'] * 100:.2f}%",
        delta=(
            f"SD ± "
            f"{evaluasi['nested_akurasi_sd'] * 100:.2f}%"
        )
    )

    n2.metric(
        "Rata-rata Macro F1",
        f"{evaluasi['nested_f1_mean'] * 100:.2f}%",
        delta=(
            f"SD ± "
            f"{evaluasi['nested_f1_sd'] * 100:.2f}%"
        )
    )

    # Evaluasi per kelas
    st.write("### Evaluasi Berdasarkan Kelas Sentimen")

    per_kelas = pd.DataFrame(
        evaluasi["per_kelas"]
    ).T

    per_kelas.index.name = "Sentimen"

    st.dataframe(
        per_kelas.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}"
        }),
        use_container_width=True
    )

    # Confusion matrix
    st.write("### Confusion Matrix")

    urutan = evaluasi["urutan_label"]

    cm = pd.DataFrame(
        evaluasi["confusion_matrix"],
        index=urutan,
        columns=urutan
    )

    cm.index.name = "Aktual"
    cm.columns.name = "Prediksi"

    st.dataframe(
        cm,
        use_container_width=True
    )

    # Parameter terbaik
    with st.expander("Lihat Parameter Terbaik Model"):

        st.json(evaluasi["parameter_terbaik"])


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Sistem Analisis Sentimen Ulasan Aplikasi Blibli | "
    "TF-IDF dan Multinomial Naive Bayes"
)

