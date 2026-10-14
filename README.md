# Sesi 1: Prediksi Mahasiswa Berisiko Drop Out (Data Publik UCI)

**Nama:** (Rheby Ersa Monica) | **NPM:** (G1A023016) | Prodi Informatika, Universitas Bengkulu

Tugas Informatics Practitioner Lecture Series, *"Make Sense of Data with Analysis and AI"* (Andria Arisal, BRIN, 5 Oktober 2026).

## Pertanyaan bisnis → pertanyaan analitik
- **Bisnis:** mahasiswa mana yang perlu didampingi lebih awal agar tidak drop out?
- **Analitik:** dari data yang tersedia pada waktu tertentu (saat daftar, akhir semester 1, akhir semester 2), seberapa baik kita memprediksi mahasiswa yang akan *Dropout*?

## Data
UCI *Predict Students' Dropout and Academic Success* (4.424 mahasiswa, 36 fitur, CC BY 4.0).
Unduh dari https://archive.ics.uci.edu/dataset/697, ekstrak zip, simpan sebagai `data/data.csv` (pemisah titik koma).

> Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Alur (CRISP-DM + 4 level analytics)
| Tahap | Isi |
|---|---|
| Data understanding | Ukuran, missing value, duplikat, distribusi target |
| Data preparation | Status *Enrolled* dikeluarkan (hasil akhir belum diketahui); fitur dipilih sesuai waktu prediksi |
| Descriptive | Tingkat dropout per kelompok (biaya kuliah, beasiswa, umur, dll.) |
| Diagnostic | Perbandingan rata-rata fitur Dropout vs Graduate |
| Predictive | Baseline, Logistic Regression, Decision Tree, Random Forest pada 3 skenario waktu |
| Prescriptive | Skor risiko → tingkat risiko → rekomendasi tindakan |

## Konsep materi yang diterapkan
- **Garbage in, garbage out / label valid:** hanya mahasiswa dengan hasil akhir yang jelas yang dipakai.
- **Data leakage (point-in-time):** fitur dibatasi sesuai kapan prediksi dibuat, sehingga tidak memakai informasi dari masa depan.
- **Data tidak seimbang:** baseline "tebak mayoritas" punya akurasi ±61% tetapi recall 0 (tidak menemukan satu pun dropout), jadi dievaluasi dengan recall, F1, dan ROC-AUC.
- **Keputusan di manusia:** target recall dan tindakan ditetapkan pengelola prodi, bukan model.

## Cara menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python case_study_uci.py
# opsi: --drop-sensitive (buang gender, kebangsaan, dll.), --include-enrolled, --recall-target 0.8
```
Hasil (grafik dan CSV) tersimpan di folder `output/`.

## Hasil (data asli UCI, seed 42)
- Data modeling: 3.630 mahasiswa (794 berstatus *Enrolled* dikeluarkan); proporsi dropout 39,1%.
- Diagnostik: pada semester 1, mahasiswa yang akhirnya dropout rata-rata hanya lulus ±2,6 mata kuliah vs ±6,2 pada yang lulus kuliah.
- Prediksi (Random Forest, ROC-AUC): saat pendaftaran ±0,86; akhir semester 1 ±0,95; akhir semester 2 ±0,97. Baseline: akurasi ±61%, recall 0.
- Fitur terpenting pada skenario B: jumlah mata kuliah lulus dan nilai semester 1, status pembayaran biaya kuliah, umur saat mendaftar, dan status beasiswa.
- Prescriptive (target recall 80%): pada data uji, tingkat dropout aktual kelompok risiko *Tinggi* ±86%, *Sedang* ±20%, *Rendah* ±4%.
- Tabel lengkap: `output/perbandingan_model.csv`; grafik: `output/*.png`.

## Keterbatasan
- Data berasal dari satu institusi di Portugal, jadi belum tentu berlaku untuk kampus lain.
- Pembagian data latih/uji acak berstrata, karena tidak ada kolom waktu.
- Beberapa fitur (misalnya status utang/biaya kuliah) bisa berubah setelah mendaftar. Pada penerapan nyata perlu dipastikan fitur tersedia saat prediksi dibuat.
- Skor skenario A kemungkinan terlalu optimis: tanpa `debtor` dan `tuition_fees_up_to_date` (yang bisa berubah setelah mendaftar), ROC-AUC Random Forest turun dari ±0,86 menjadi ±0,80.
- Fitur sensitif sebaiknya tidak dipakai menentukan tindakan terhadap individu (gunakan `--drop-sensitive`).
