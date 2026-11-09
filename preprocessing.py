"""
preprocessing.py
Fungsi pra-pemrosesan teks yang dipakai BERSAMA oleh notebook pelatihan dan aplikasi Streamlit
(sesuai skripsi Subbab 4.1.1 dan Lampiran 2A: fungsi yang sama dipakai di pelatihan dan di sistem).

Tahapan: case folding -> cleaning (URL, mention, tagar, tanda baca, angka berdiri sendiri)
         -> tokenisasi sederhana -> stopword removal (kata negasi DIPERTAHANKAN).
Stemming TIDAK dipakai.
"""
import re

# Urutan kelas model (sama dengan urutan alfabet yang dipakai scikit-learn)
LABELS = ["Negatif", "Netral", "Positif"]

PESAN_TIDAK_INFORMATIF = "Teks tidak cukup informatif"

# Kata negasi: SENGAJA tidak dibuang karena bisa membalik makna kalimat
NEGASI = {
    "tidak", "tak", "gak", "ga", "nggak", "ngga", "enggak", "bukan", "belum",
    "kurang", "jangan", "tanpa", "not", "no", "never",
}

# Daftar stopword bahasa Indonesia (+ sedikit bahasa Inggris) buatan sendiri.
# CATATAN: kalau kamu sudah punya daftar stopword asli yang dipakai di skripsi,
# GANTI isi daftar ini dengan daftar aslimu supaya hasilnya sama persis.
_STOPWORDS_TEXT = """
yang dan di ke dari untuk dengan pada ini itu atau karena juga saya aku kami kita
mereka anda kamu dia ia adalah ada akan agar oleh dalam sudah telah sedang lagi
bisa dapat hanya saja sampai hingga sebagai sebuah suatu para tersebut bahwa jadi
maka namun serta pun lah kah nya sih deh dong kok nih tuh ya yah dgn dg utk dr krn
sy kalo kalau jika apabila bila sebelum sesudah setelah ketika saat
the a an and is are was were to of it i for in this that you my with be so am
"""
STOPWORDS_ID = set(_STOPWORDS_TEXT.split()) - NEGASI


def preprocessing(teks):
    """Membersihkan satu teks ulasan. Mengembalikan teks bersih (string, bisa kosong)."""
    teks = str(teks).lower()                                  # 1. case folding
    teks = re.sub(r"http\S+|www\S+", " ", teks)               # 2. cleaning: URL
    teks = re.sub(r"@\w+", " ", teks)                         #    mention
    teks = re.sub(r"#", "", teks)                             #    tanda pagar
    teks = re.sub(r"[^a-z0-9\s]", " ", teks)                  #    tanda baca, simbol, emoji
    teks = re.sub(r"\b\d+\b", " ", teks)                      #    angka yang berdiri sendiri
    teks = re.sub(r"\s+", " ", teks).strip()                  #    rapikan spasi
    tokens = teks.split()                                     # 3. tokenisasi sederhana
    tokens = [k for k in tokens                               # 4. stopword removal
              if k not in STOPWORDS_ID and len(k) > 1]
    return " ".join(tokens)


def tahapan_preprocessing(teks):
    """Dipakai untuk membuat contoh Tabel 4.2 (hasil pada setiap tahap)."""
    asli = str(teks)
    folding = asli.lower()
    t = re.sub(r"http\S+|www\S+", " ", folding)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    cleaning = re.sub(r"\s+", " ", t).strip()
    tokens = cleaning.split()
    akhir = [k for k in tokens if k not in STOPWORDS_ID and len(k) > 1]
    return {"Ulasan asli": asli, "1. Case folding": folding, "2. Cleaning": cleaning,
            "3. Tokenisasi": tokens, "4. Stopword removal (hasil akhir)": akhir}
