"""
1_Input.py
==========
Halaman Input — User mengisi profil untuk simulasi skill gap.
"""

import streamlit as st
import os
import sys
import json
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Input Profil | Simulator Kesenjangan Keahlian", page_icon=None, layout="wide")

# Inject Custom CSS for Premium Slate/Teal Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply Inter font globally to text containers, avoiding icon spans */
    html, body, p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stSelectbox, .stSlider, .stButton {
        font-family: 'Inter', sans-serif !important;
    }

    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #e2e8f0 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d9488 0%, #115e59 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.25);
    }
    [data-testid="stMetric"] label {
        color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load skill master
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
df_skill_master = pd.read_csv(os.path.join(PROCESSED_DIR, "skill_master_cleaned.csv"))

# ============================================================
# SKILL DEFAULTS BY BACKGROUND
# ============================================================
BACKGROUND_DEFAULTS = {
    "Pemula": {"mean": 25, "soft_bonus": 10},
    "Menengah": {"mean": 55, "soft_bonus": 5},
    "Lanjutan": {"mean": 78, "soft_bonus": 3},
}

# ============================================================
# PAGE CONTENT
# ============================================================
st.markdown("# Input Profil Anda")
st.markdown("Isi profil di bawah ini untuk mendapatkan analisis kesenjangan keahlian dan rencana belajar personal.")

# Guide Expander for New Users
with st.expander("Panduan Pengguna: Cara Mengisi Skor & Memilih Skenario"):
    col_guide1, col_guide2 = st.columns(2)
    with col_guide1:
        st.markdown("""
        **Panduan Penilaian Mandiri Keahlian (Skala 0 - 100):**
        * **0 - 20 (Pemula / Beginner)**: Hanya memahami teori dasar, belum pernah mempraktikkannya secara nyata.
        * **21 - 50 (Menengah-Bawah / Advanced Beginner)**: Bisa mempraktikkan keahlian dasar jika dibantu tutorial/panduan.
        * **51 - 75 (Menengah-Atas / Intermediate)**: Mampu mengerjakan proyek secara mandiri tanpa tutorial terus-menerus.
        * **76 - 100 (Ahli / Advanced - Expert)**: Sangat mahir, mampu mendesain sistem kompleks atau mengoptimalkan kode.
        
        *Tip: Anda juga bisa mengklik tombol **Isi Otomatis** di bawah untuk estimasi nilai awal.*
        """)
    with col_guide2:
        st.markdown("""
        **Panduan Pemilihan Skenario Pasar Kerja:**
        * **Normal**: Kondisi pasar stabil. Standar penyaringan perusahaan dan estimasi waktu siap kerja berjalan normal.
        * **Kompetitif**: Pelamar melimpah/lowongan sedikit (persaingan ketat). Standar rekrutmen naik dan waktu belajar lebih lama.
        * **Booming**: Banyak lowongan baru dibuka (kebutuhan tinggi). Perusahaan mempermudah rekrutmen dan waktu belajar lebih singkat.
        """)

st.markdown("---")

# ---- Section 1: Basic Info ----
st.markdown("### Informasi Dasar")

col1, col2 = st.columns(2)

with col1:
    target_role = st.selectbox(
        "Target Karier",
        ["Data Analyst", "Frontend Developer", "UI/UX Designer"],
        help="Pilih bidang karier yang Anda tuju"
    )
    
    background_level = st.selectbox(
        "Tingkat Latar Belakang",
        ["Pemula", "Menengah", "Lanjutan"],
        help="Pemula: baru mulai belajar | Menengah: sudah ada dasar | Lanjutan: sudah berpengalaman"
    )

with col2:
    study_hours = st.slider(
        "Jam Belajar per Minggu",
        min_value=3, max_value=25, value=10, step=1,
        help="Berapa jam per minggu yang bisa Anda dedikasikan untuk belajar?"
    )
    
    market_scenario = st.selectbox(
        "Skenario Pasar Kerja",
        ["Normal", "Kompetitif", "Booming"],
        help="Normal: standar | Kompetitif: persaingan ketat (+30% waktu) | Booming: banyak lowongan (-20% waktu)"
    )

st.markdown("---")

# ---- Section 2: Skill Proficiency ----
st.markdown("### Tingkat Kemahiran Keahlian Anda")
st.markdown(f"Atur tingkat kemahiran Anda untuk setiap keahlian **{target_role}** (0 = tidak menguasai, 100 = sangat mahir).")

# Get skills for selected role
role_skills = df_skill_master[df_skill_master['role'] == target_role].sort_values(
    'importance_score', ascending=False
)

# Auto-fill button - write directly to session state keys
bg_defaults = BACKGROUND_DEFAULTS[background_level]
if st.button(f"Isi Otomatis Berdasarkan Tingkat '{background_level}'", type="secondary"):
    for _, row in role_skills.iterrows():
        key = f"skill_{row['skill_name']}"
        val_default = bg_defaults['mean']
        if row['skill_category'] == "Soft Skill":
            val_default = min(100, val_default + bg_defaults['soft_bonus'])
        st.session_state[key] = int(val_default)

skill_proficiencies = {}

# Group by category
categories = role_skills['skill_category'].unique()
tabs = st.tabs([c for c in categories])

for tab, category in zip(tabs, categories):
    with tab:
        cat_skills = role_skills[role_skills['skill_category'] == category]
        
        cols = st.columns(2)
        for i, (_, row) in enumerate(cat_skills.iterrows()):
            with cols[i % 2]:
                # Use session state for binding
                key = f"skill_{row['skill_name']}"
                
                # Set initial default to 50 if key does not exist yet
                if key not in st.session_state:
                    st.session_state[key] = 50
                
                val = st.slider(
                    f"{row['skill_name']} (Importance Score: {row['importance_score']:.2f})",
                    min_value=0, max_value=100,
                    key=key,
                    help=f"Importance Score: {row['importance_score']:.2f} | "
                         f"Rata-rata waktu belajar: {row['avg_learning_hours']} jam"
                )
                skill_proficiencies[row['skill_name']] = val

st.markdown("---")

# ---- Section 3: Submit ----
st.markdown("### Jalankan Simulasi")

col_submit1, col_submit2 = st.columns([3, 1])

with col_submit1:
    st.info(f"""
    **Ringkasan Informasi:**
    - Target Karier: **{target_role}**
    - Tingkat Latar Belakang: **{background_level}**
    - Alokasi Belajar: **{study_hours} jam/minggu**
    - Skenario Pasar: **{market_scenario}**
    - Jumlah Keahlian Diisi: **{len(skill_proficiencies)} keahlian**
    """)

with col_submit2:
    if st.button("Mulai Simulasi", type="primary", use_container_width=True):
        # Store in session state for other pages
        st.session_state['input_data'] = {
            'target_role': target_role,
            'background_level': background_level,
            'study_hours_per_week': study_hours,
            'market_scenario': market_scenario,
            'skill_proficiencies': skill_proficiencies,
        }
        st.session_state['simulation_ready'] = True
        st.success("Data tersimpan! Buka halaman Hasil Simulasi untuk melihat hasil.")
        st.balloons()

# Show current session state status
if st.session_state.get('simulation_ready'):
    st.success("Data sudah siap. Navigasi ke halaman Hasil Simulasi di sidebar.")
