# Data Dictionary — Capstone Project CC26-PRU469
## Skill Gap Simulator Dengan Career Scenario Engine

**Tanggal**: 29 Mei 2026  
**Sumber Data**: Kaggle (1.3M LinkedIn Jobs & Skills 2024) + Sintetis  
**Referensi Dataset**: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024

---

## 1. job_postings_cleaned.csv (600 rows)

| Kolom | Tipe | Deskripsi | Contoh |
|---|---|---|---|
| job_id | string | ID unik job posting | JOB-00001 |
| job_title | string | Judul pekerjaan | Senior Data Analyst |
| company | string | Nama perusahaan | Google |
| job_location | string | Lokasi pekerjaan | Jakarta, Indonesia |
| role_category | string | Kategori role target (3 nilai) | Data Analyst |
| job_level | string | Level pekerjaan | Entry level / Mid-Senior |
| job_type | string | Tipe pekerjaan | Full-time / Contract |
| first_seen | datetime | Tanggal posting pertama kali terlihat | 2024-03-15 |
| source | string | Sumber data | Kaggle-LinkedIn-Synthetic |
| month | int | Bulan posting (1-12) | 3 |
| quarter | int | Kuartal posting (1-4) | 1 |

---

## 2. job_skills_cleaned.csv (~6,936 rows)

| Kolom | Tipe | Deskripsi | Contoh |
|---|---|---|---|
| job_id | string | FK ke job_postings | JOB-00001 |
| skill_name | string | Nama skill yang dibutuhkan | Python |
| skill_category | string | Kategori skill | Technical / Tool / Soft Skill |
| role_category | string | Role target | Data Analyst |

---

## 3. skill_master_cleaned.csv (60 rows)

| Kolom | Tipe | Deskripsi | Range |
|---|---|---|---|
| role | string | Role target | 3 nilai |
| skill_name | string | Nama skill | 20 per role |
| skill_category | string | Kategori | Technical / Tool / Soft Skill |
| importance_score | float | Skor kepentingan skill | 0.0 - 1.0 |
| avg_learning_hours | int | Estimasi jam belajar dari 0 ke kompeten | 20 - 250 |
| market_demand_normal | float | Bobot di skenario Normal | 0.0 - 1.0 |
| market_demand_competitive | float | Bobot di skenario Kompetitif | 0.0 - 1.0 |
| market_demand_booming | float | Bobot di skenario Booming | 0.0 - 1.0 |

---

## 4. user_profiles_cleaned.csv (750 rows)

| Kolom | Tipe | Deskripsi | Range/Contoh |
|---|---|---|---|
| user_id | string | ID unik user sintetis | USR-0001 |
| target_role | string | Role yang dituju | Data Analyst |
| background_level | string | Level background user | Pemula / Menengah / Lanjutan |
| study_hours_per_week | float | Jam belajar per minggu | 3.0 - 25.0 |
| market_scenario | string | Skenario pasar | Normal / Kompetitif / Booming |
| gap_score | float | Weighted Gap Score (computed) | 0.0 - 1.0 |
| readiness_label | string | Label kesiapan | Ready / Almost Ready / Needs Work / Significant Gap / Major Gap |
| top_missing_skill_1 | string | Missing skill #1 | SQL |
| top_missing_skill_2 | string | Missing skill #2 | Python |
| top_missing_skill_3 | string | Missing skill #3 | Statistics |
| estimated_weeks_ready | float | Estimasi minggu siap kerja | varies |
| current_skills_json | string (JSON) | Proficiency semua skill | {"Python": 45, "SQL": 30, ...} |

---

## 5. user_skills_cleaned.csv (15,000 rows)

| Kolom | Tipe | Deskripsi | Range |
|---|---|---|---|
| user_id | string | FK ke user_profiles | USR-0001 |
| target_role | string | Role target user | Data Analyst |
| skill_name | string | Nama skill | Python |
| proficiency_level | int | Level proficiency user | 0 - 100 |
| background_level | string | Background user | Pemula |
| market_scenario | string | Skenario pasar | Normal |

---

## 6. skill_frequency.csv

| Kolom | Tipe | Deskripsi |
|---|---|---|
| role_category | string | Role target |
| skill_name | string | Nama skill |
| frequency | int | Jumlah job posting yang membutuhkan skill ini |
| total_postings | int | Total job posting untuk role ini |
| frequency_pct | float | Persentase frekuensi (%) |

---

## 7. modeling_dataset.csv (750 rows, ~60+ columns)

Dataset gabungan yang siap dipakai untuk modeling dan dashboard.
Berisi metadata user + proficiency per skill dalam format wide (1 kolom per skill).

---

## Formula

### Weighted Gap Score
```
Gap Score = 1 - (SUM(user_skill_i * importance_i) / SUM(importance_i))
```

### Estimasi Waktu Siap Kerja
```
Weeks = SUM(gap_i * avg_learning_hours_i * importance_i) / study_hours_per_week * multiplier
```
- Normal: multiplier = 1.0
- Kompetitif: multiplier = 1.3
- Booming: multiplier = 0.8

### Readiness Labels
| Gap Score | Label |
|---|---|
| 0.0 - 0.2 | Ready |
| 0.2 - 0.4 | Almost Ready |
| 0.4 - 0.6 | Needs Work |
| 0.6 - 0.8 | Significant Gap |
| 0.8 - 1.0 | Major Gap |
