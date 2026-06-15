# Air Quality Analytics — Urban Environment

Dashboard demo Tugas Besar **II4013 Data Analytics (Kelompok 4)** dengan topik
*Kualitas Udara dan Lingkungan Perkotaan*. Aplikasi ini memvisualisasikan hasil
analisis kualitas udara global dan menyediakan simulasi prediksi AQI, sebagai
implementasi sistem (tahap iNterpret pada kerangka OSEMN).

## Sumber Data

| Dataset | Peran | Baris × Kolom |
|---|---|---|
| OpenAQ Global Air Quality | Utama (polutan, timestamp, koordinat) | 57.815 × 12 |
| Global Air Pollution (Kaggle) | Utama (AQI Value & kategori per polutan) | 23.463 × 12 |
| Global Air Quality Dataset | Pendukung (suhu, kelembapan, angin) | 10.000 × 12 |

Dataset utama digabung dengan *inner join* pada pasangan kota–negara, menghasilkan
**14.524 baris × 22 kolom** (33 negara, 386 kota) yang menjadi data inti dashboard.

## Struktur Folder

```
TubesDatalVisualisasi/
├── app.py                              aplikasi Streamlit (6 halaman)
├── prepare_data.py                     pipeline data + pelatihan model
├── requirements.txt                    daftar dependensi
├── README.md                           petunjuk penggunaan
├── data/
│   ├── raw/                            2 dataset mentah
│   └── processed/                      dataset hasil scrub (dengan dan tanpa encode)
├── models/
│   ├── aqi_model.pkl                   model terbaik + daftar fitur
│   └── metrics.json                    metrik & feature importance
└── II4013_Notebook_Kelompok_4.ipynb    notebook ipynb
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python prepare_data.py          # menyiapkan data bersih dan melatih model
streamlit run app.py
```

Jika folder `data/raw/` kosong, `prepare_data.py` akan mengunduh dataset dari
Google Drive secara otomatis (membutuhkan koneksi internet).

## Halaman Dashboard

1. **Overview / KPI** — ringkasan total record, cakupan wilayah, rata-rata AQI, proporsi Hazardous, distribusi kategori, dan tren bulanan.
2. **Eksplorasi Tren** — pola AQI per bulan dan per jam dengan filter negara dan rentang bulan.
3. **Perbandingan Wilayah** — peta sebaran AQI per kota serta peringkat negara terbersih dan tercemar.
4. **Analisis Polutan** — kontribusi tiap polutan, korelasi antarvariabel, sebaran per kategori, dan pengaruh faktor cuaca.
5. **Prediksi / Model** — perbandingan performa model, feature importance, dan simulasi prediksi AQI interaktif.
6. **Insight & Rekomendasi** — temuan utama dan rekomendasi tindakan beserta pihak sasaran dan prioritas.

## Catatan Model

Model regresi AQI bersifat **explanatory**. Sub-indeks polutan (terutama PM2.5)
berhubungan langsung dengan AQI Value sehingga R² mendekati 1. Model digunakan
untuk mengidentifikasi faktor dominan pembentuk AQI, bukan sebagai sistem prediksi
operasional yang independen dari komponen AQI.
