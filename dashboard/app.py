"""
app.py
======
Main Streamlit Application — Skill Gap Simulator Dashboard
3 halaman: Input → Hasil Simulasi → Rencana Aksi
"""

import streamlit as st
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Simulator Kesenjangan Keahlian",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
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
    
    /* Header styling */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Phase cards */
    .phase-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #0d9488;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        color: #f8fafc;
    }
    
    /* Skill bar */
    .skill-bar-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 20px;
        margin: 5px 0;
        position: relative;
    }
    .skill-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# Simulator Kesenjangan Keahlian")
    st.markdown("### Simulasi Skenario Karier")
    st.markdown("---")
    st.markdown("""
    **Proyek Capstone CC26-PRU469**  
    Coding Camp 2026 — DBS Foundation
    
    ---
    
    **Cara Penggunaan:**
    1. Isi profil di halaman **Input**
    2. Lihat hasil di **Hasil Simulasi**
    3. Ikuti **Rencana Aksi**
    
    ---
    
    *Dukung oleh TensorFlow & Streamlit*
    """)

# ============================================================
# HOME PAGE
# ============================================================
st.markdown('<p class="hero-title">Simulator Kesenjangan Keahlian</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Dengan Simulasi Skenario Karier — Temukan celah keahlian Anda dan dapatkan rencana belajar personal</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Langkah 1: Input Profil
    Pilih karier sasaran, tentukan tingkat kemahiran saat ini, 
    alokasikan jam belajar per minggu, dan tentukan skenario pasar.
    
    Buka halaman **1_Input** di sidebar untuk memulai.
    """)

with col2:
    st.markdown("""
    ### Langkah 2: Hasil Simulasi
    Tinjau **Skor Kesenjangan**, **Keahlian Paling Dibutuhkan**, 
    **Estimasi Waktu Siap Kerja**, serta 
    **Perbandingan Kondisi Pasar**.
    
    Buka halaman **2_Hasil_Simulasi** untuk melihat grafik.
    """)

with col3:
    st.markdown("""
    ### Langkah 3: Rencana Aksi
    Dapatkan **kurikulum belajar** personal berbasis kecerdasan buatan:
    tahapan materi belajar, jadwal mingguan, pencapaian target, 
    serta rekomendasi sumber belajar.
    
    Buka halaman **3_Rencana_Aksi** untuk memulai pembelajaran.
    """)

st.markdown("---")

# Quick stats
st.markdown("### Tentang Sistem")
stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Karier Target", "3", help="Data Analyst, Frontend Developer, UI/UX Designer")
stat2.metric("Keahlian Dinilai", "60", help="20 keahlian unik per bidang karier")
stat3.metric("Skenario Pasar", "3", help="Normal, Kompetitif, Booming")
stat4.metric("Model AI", "Multi-Output", help="TensorFlow Functional API")

st.markdown("---")
st.caption("Proyek Capstone CC26-PRU469 | Coding Camp 2026 didukung oleh DBS Foundation | Tema: Future-Ready Work & Economy")
