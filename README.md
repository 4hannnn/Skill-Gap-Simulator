# Skill Gap Simulator dengan Career Scenario Engine

> **Capstone Project ID**: CC26-PRU469
> **Tema**: Future-Ready Work & Economy
> **Coding Camp 2026** powered by **DBS Foundation**

---

## Deskripsi Proyek

**Skill Gap Simulator** adalah sistem berbasis kecerdasan buatan yang dirancang untuk membantu *fresh graduate*, *career switcher*, dan pencari kerja pemula mengukur kesiapan kerja mereka secara kuantitatif. Sistem ini memprediksi seberapa besar kesenjangan (*gap*) antara profil keahlian pengguna saat ini dengan standar yang dibutuhkan industri, mengestimasi waktu yang diperlukan untuk siap bekerja, dan menghasilkan kurikulum belajar personal yang adaptif terhadap kondisi pasar kerja.

Solusi akhir proyek ini berupa **Dashboard Interaktif Streamlit** 3 halaman yang terhubung langsung dengan model **Deep Learning TensorFlow Functional API** siap produksi.

---

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Weighted Gap Score** | Kalkulasi kesenjangan kompetensi menggunakan pembobotan *importance score* tiap skill |
| **Readiness Label** | Klasifikasi ke 5 kategori: Ready, Almost Ready, Needs Work, Significant Gap, Major Gap |
| **Top 3 Missing Skills** | Identifikasi 3 skill kritis yang paling perlu ditingkatkan |
| **Estimasi Waktu Siap Kerja** | Prediksi durasi (minggu) berdasarkan jam belajar dan tingkat keparahan gap |
| **Career Scenario Engine** | Simulasi 3 kondisi pasar: Normal, Kompetitif (+30%), Booming (-20%) |
| **Roadmap Belajar AI** | Fase belajar modular: Foundation → Deepening → Polish + jadwal mingguan |

---

## Arsitektur Sistem

```
Input Profil User
(Role, Background, Skill Proficiency, Jam Belajar, Skenario)
        |
        v
┌─────────────────────────────────────────────┐
│         Model Deep Learning (TensorFlow)    │
│         Functional API — Multi-Output       │
│                                             │
│  Input Layer (60+ fitur)                   │
│       → Shared Backbone (Dense 128→64)     │
│       → Output 1: Gap Score (Regresi)      │
│       → Output 2: Readiness (Klasifikasi)  │
│       → Output 3: Weeks Ready (Regresi)    │
│                                             │
│  Custom Components:                         │
│  - WeightedGapLoss (Custom Loss)            │
│  - EarlyStoppingWithRestore (Custom CB)     │
└─────────────────────────────────────────────┘
        |
        v
Dashboard Streamlit 3 Halaman:
  1_Input → 2_Hasil_Simulasi → 3_Rencana_Aksi
```

---

## Performa Model

| Metrik | Nilai |
|---|---|
| Gap Score MAE | 0.0136 |
| Gap Score RMSE | 0.0176 |
| Readiness Accuracy | **95.2%** |
| Weeks MAE | 4.4 minggu |
| Test Samples | 900 |
| Best Epoch | 91 / 106 |

---

## Struktur Repositori

```
Capstone_Project/
├── dashboard/                        # Aplikasi web Streamlit
│   ├── app.py                        # Halaman beranda
│   └── pages/
│       ├── 1_Input.py                # Form input profil & proficiency skill
│       ├── 2_Hasil_Simulasi.py       # Visualisasi hasil (gauge, radar, bar)
│       └── 3_Rencana_Aksi.py         # Roadmap belajar berbasis AI
│
├── data/
│   ├── raw/                          # Dataset mentah (Kaggle LinkedIn + sintetis)
│   └── processed/                    # Dataset bersih & siap modeling
│       ├── job_postings_cleaned.csv  # 600 job posting (3 role)
│       ├── job_skills_cleaned.csv    # ~6.936 skill requirements
│       ├── skill_master_cleaned.csv  # 60 skill (20 per role) + importance score
│       ├── user_profiles_cleaned.csv # 750 profil user sintetis + gap score
│       ├── user_skills_cleaned.csv   # 15.000 baris proficiency per skill
│       ├── skill_frequency.csv       # Frekuensi skill per role
│       └── modeling_dataset.csv      # Dataset gabungan wide-format (750 x 60+)
│
├── docs/
│   └── data_dictionary.md            # Kamus data lengkap semua dataset + formula
│
├── models/
│   └── skill_gap_model/
│       ├── skill_gap_model.keras     # Model TensorFlow siap produksi (330 KB)
│       ├── model_metadata.json       # Kolom fitur, encoder, metrik evaluasi
│       └── training_history.json     # Riwayat loss/accuracy per epoch
│
├── notebooks/
│   ├── 01_data_wrangling.ipynb       # Gathering, Assessing, Cleaning data
│   ├── 02_eda.ipynb                  # EDA + 11 visualisasi + interpretasi
│   ├── 03_explanatory_analysis.ipynb # Jawaban visual atas Research Questions
│   └── 04_skill_matching_gap_scoring.ipynb  # Formula gap scoring
│
├── reports/                          # Grafik EDA & Explanatory Analysis (.png)
│
├── scripts/
│   ├── cleaning/
│   │   └── run_cleaning_pipeline.py  # Pipeline pembersihan data end-to-end
│   ├── model/
│   │   ├── train_model.py            # Training model + custom loss & callback
│   │   ├── inference.py              # Kode inferensi siap produksi
│   │   └── learning_strategy.py      # Generator roadmap belajar personal
│   └── synth/
│       └── generate_user_profiles.py # Generator data sintetis user
│
├── Project Plan/
│   └── Project_Plan_Filled.md        # Dokumen rencana & metodologi proyek
│
├── requirements.txt                  # Dependensi Python
├── run_app.bat                       # Shortcut jalankan Streamlit (Windows)
└── README.md
```

---

## Dataset

Seluruh dataset yang digunakan dalam proyek ini dibuat secara **sintetis (generated)** dengan pemodelan distribusi yang didasarkan pada pola data industri riil:

**1. Data Job Postings & Skill Master (Pemodelan Pola Industri)**
* **Dasar Pola**: Pola data diselaraskan dengan frekuensi dan tingkat kepentingan keahlian pada dataset publik [1.3M LinkedIn Jobs & Skills 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) oleh asaniczka.
* **Metode**: Dibuat menggunakan skrip generator ([`generate_job_postings.py`](./scripts/synth/generate_job_postings.py)) untuk menghasilkan 600 lowongan kerja beserta kebutuhan keahliannya bagi 3 target role, lengkap dengan data kotor (*dirty data* seperti typo, *missing values*, duplikasi) untuk simulasi prapemrosesan riil.

**2. Data Profil Pengguna (Sintetis)**
* **Metode**: Dibuat menggunakan skrip generator ([`generate_user_profiles.py`](./scripts/synth/generate_user_profiles.py)) untuk menghasilkan 4.500 profil pengguna beserta tingkat kemahiran (*proficiency*) masing-masing skill.
* **Kalkulasi**: Skor kesenjangan (*gap score*), tingkat kesiapan (*readiness label*), dan estimasi waktu siap kerja dihitung secara deterministik menggunakan formula bisnis tanpa kebocoran data (*anti-data leakage*).

**Formula Gap Score:**
```
Gap Score = 1 - SUM(proficiency_i × importance_i) / SUM(importance_i)
```
*Nilai 0.0 = siap kerja, nilai 1.0 = gap maksimal*

---

## Model AI

Model dibangun menggunakan **TensorFlow Functional API** dengan arsitektur multi-output:

```python
inputs = keras.Input(shape=(n_features,))
# Shared backbone
x = Dense(128, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)

# Output 1: Gap Score (regresi)
gap_output = Dense(1, activation='sigmoid', name='gap_output')(...)

# Output 2: Readiness (klasifikasi 5 kelas)
readiness_output = Dense(5, activation='softmax', name='readiness_output')(...)

# Output 3: Estimated Weeks (regresi)
weeks_output = Dense(1, activation='relu', name='weeks_output')(...)

model = Model(inputs=inputs, outputs=[gap_output, readiness_output, weeks_output])
```

**Custom Components yang Diimplementasikan:**

1. **`WeightedGapLoss`** (Custom Loss Function)
   - Memberi bobot lebih tinggi pada error di area gap kritis (gap > 0.6)
   - Formula: `loss = mean(weight × (y_true - y_pred)²)`

2. **`EarlyStoppingWithRestore`** (Custom Callback)
   - Monitor multi-metrik: `-val_loss + 0.3 × val_readiness_accuracy`
   - Otomatis restore best weights saat training berhenti

---

## Tautan & Cara Memuat Model ML

### 1. Tautan Unduh Model
Model ML hasil pelatihan telah disertakan secara langsung di dalam direktori repositori/ZIP ini pada path:
* **[models/skill_gap_model/skill_gap_model.keras](./models/skill_gap_model/skill_gap_model.keras)** (tersedia secara lokal).

Jika Anda ingin mengunduhnya secara terpisah, model dapat diunduh dari tautan raw repositori GitHub Anda:
* **Tautan Unduh**: `https://github.com/[USERNAME]/[REPO_NAME]/raw/main/models/skill_gap_model/skill_gap_model.keras` (ganti `[USERNAME]` dan `[REPO_NAME]` dengan username dan nama repositori Anda setelah di-push).

### 2. Cara Memuat (Load) Model
Model disimpan menggunakan format `.keras` penuh. Karena model menggunakan komponen kustom (`WeightedGapLoss`), gunakan argumen `custom_objects` saat memuat model:

```python
import tensorflow as tf
from tensorflow import keras
from scripts.model.train_model import WeightedGapLoss

# Memuat model
model = keras.models.load_model(
    'models/skill_gap_model/skill_gap_model.keras',
    custom_objects={'WeightedGapLoss': WeightedGapLoss}
)
```

---

## Cara Menjalankan

### Prasyarat
- Python 3.11+
- Virtual environment aktif
- Berkas template environment **`.env.example`** disediakan di root direktori (proyek ini berjalan 100% secara lokal tanpa API eksternal, sehingga tidak memerlukan kredensial khusus di `.env`).

### 1. Clone & Setup Environment
```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi (Windows)
.venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

### 2. Jalankan Dashboard (Cara Mudah)
Double-click file `run_app.bat` di File Explorer.

**Atau via terminal:**
```bash
.venv\Scripts\streamlit.exe run dashboard/app.py
```

Buka browser di: **http://localhost:8501**

> **PENTING**: Jangan gunakan `streamlit run` langsung tanpa prefix `.venv\Scripts\` karena model `.keras` harus dimuat dengan Python versi yang sama saat training (Python 3.11).

### 3. Melatih Ulang Model (Opsional)
```bash
.venv\Scripts\python.exe scripts/model/train_model.py
```

### 4. Uji Inferensi via Terminal
```bash
.venv\Scripts\python.exe scripts/model/inference.py
```

---

## Cara Penggunaan Aplikasi

1. **Halaman Input** (`1_Input`)
   - Pilih role target: Data Analyst / Frontend Developer / UI/UX Designer
   - Tentukan background level: Pemula / Menengah / Lanjutan
   - Tentukan skenario pasar: Normal / Kompetitif / Booming
   - Atur jam belajar per minggu (3–25 jam)
   - Input proficiency untuk setiap skill (slider 0–100)
   - Klik **"Simulasikan"**

2. **Halaman Hasil Simulasi** (`2_Hasil_Simulasi`)
   - Lihat **Skill Gap Score** via gauge chart
   - Cek **Readiness Label** dan top 3 missing skills
   - Bandingkan estimasi waktu di 3 skenario pasar berbeda

3. **Halaman Rencana Aksi** (`3_Rencana_Aksi`)
   - Ikuti roadmap belajar 3 fase yang dipersonalisasi
   - Lihat jadwal mingguan dan sumber belajar rekomendasi
   - Pantau target milestone progress

---

## Tim Capstone CC26-PRU469

| Nama | Role | Tanggung Jawab |
|---|---|---|
| **Awanda Rosi Firdaus** | AI Engineer | Model TensorFlow, Custom Loss, Inference, Dashboard |
| **Muhammad Farhan Jayandra** | Data Scientist | Data Wrangling, EDA, Analisis Bisnis, Visualisasi |
| **Claudio Amadeus Modena** | Data Scientist | Data Wrangling, EDA, Analisis Bisnis, Visualisasi |

---

*Proyek Capstone CC26-PRU469 | Coding Camp 2026 didukung oleh DBS Foundation | Tema: Future-Ready Work & Economy*
