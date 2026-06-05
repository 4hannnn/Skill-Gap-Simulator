"""
2_Hasil_Simulasi.py
===================
Halaman Hasil Simulasi — Menampilkan 4 output inti KPI:
1. Gap Score (gauge chart)
2. Top 3 Missing Skills
3. Prediksi Waktu Siap Kerja
4. Career Scenario Comparison
"""

import streamlit as st
import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Hasil Simulasi | Simulator Kesenjangan Keahlian", page_icon=None, layout="wide")

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

# ============================================================
# CHECK INPUT DATA
# ============================================================
if not st.session_state.get('simulation_ready'):
    st.warning("Belum ada data input. Silakan isi profil di halaman Input terlebih dahulu.")
    st.page_link("pages/1_Input.py", label="Buka Halaman Input", icon=None)
    st.stop()

input_data = st.session_state['input_data']

# ============================================================
# LOAD MODEL & PREDICT
# ============================================================
@st.cache_resource
def load_model():
    """Load model sekali saja."""
    try:
        from scripts.model.inference import load_model_and_metadata
        model, metadata, df_skill_master = load_model_and_metadata()
        return model, metadata, df_skill_master, True
    except Exception as e:
        import traceback
        traceback.print_exc()
        import keras as k
        import sys as s
        st.warning(f"Model belum tersedia ({str(e)[:80]}...). Keras: {k.__version__}, Python: {s.version.split()[0]}. Menggunakan formula deterministik.")
        PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
        df_skill_master = pd.read_csv(os.path.join(PROCESSED_DIR, "skill_master_cleaned.csv"))
        return None, None, df_skill_master, False


def deterministic_predict(input_data, df_skill_master):
    """Fallback: prediksi menggunakan formula deterministik (tanpa model)."""
    
    role = input_data['target_role']
    scenario = input_data['market_scenario']
    study_hours = input_data['study_hours_per_week']
    skills = input_data['skill_proficiencies']
    
    role_skills = df_skill_master[df_skill_master['role'] == role]
    
    multiplier_gap = {'Normal': 1.0, 'Kompetitif': 1.15, 'Booming': 0.90}
    multiplier_weeks = {'Normal': 1.0, 'Kompetitif': 1.3, 'Booming': 0.8}
    
    m_gap = multiplier_gap[scenario]
    m_weeks = multiplier_weeks[scenario]
    
    weighted_sum = 0
    importance_sum = 0
    total_hours = 0
    skill_gaps = []
    
    for _, row in role_skills.iterrows():
        imp = min(1.0, row['importance_score'] * m_gap)
        user_level = skills.get(row['skill_name'], 0) / 100.0
        
        weighted_sum += user_level * imp
        importance_sum += imp
        
        gap = max(0, 1 - user_level)
        total_hours += gap * row['avg_learning_hours'] * row['importance_score']
        
        skill_gaps.append({
            'skill_name': row['skill_name'],
            'category': row['skill_category'],
            'proficiency': skills.get(row['skill_name'], 0),
            'importance': row['importance_score'],
            'gap': round(row['importance_score'] * (1 - user_level), 4),
            'avg_learning_hours': int(row['avg_learning_hours']),
        })
    
    gap_score = round(1 - (weighted_sum / importance_sum), 4) if importance_sum > 0 else 1.0
    est_weeks = round((total_hours / study_hours) * m_weeks, 1) if study_hours > 0 else 999
    
    # Readiness label
    if gap_score <= 0.2: readiness = "Ready"
    elif gap_score <= 0.4: readiness = "Almost Ready"
    elif gap_score <= 0.6: readiness = "Needs Work"
    elif gap_score <= 0.8: readiness = "Significant Gap"
    else: readiness = "Major Gap"
    
    skill_gaps.sort(key=lambda x: x['gap'], reverse=True)
    
    return {
        'gap_score': gap_score,
        'readiness_label': readiness,
        'estimated_weeks': est_weeks,
        'top_missing_skills': skill_gaps[:3],
        'all_skill_gaps': skill_gaps,
        'input_summary': input_data,
    }


# Load model
model, metadata, df_skill_master, model_loaded = load_model()

# Predict
if model_loaded:
    from scripts.model.inference import predict_user, predict_all_scenarios
    result = predict_user(
        input_data['target_role'], input_data['background_level'],
        input_data['market_scenario'], input_data['study_hours_per_week'],
        input_data['skill_proficiencies'],
        model, metadata, df_skill_master,
    )
    # All scenarios
    scenario_results = predict_all_scenarios(
        input_data['target_role'], input_data['background_level'],
        input_data['study_hours_per_week'], input_data['skill_proficiencies'],
        model, metadata, df_skill_master,
    )
else:
    result = deterministic_predict(input_data, df_skill_master)
    # All scenarios (deterministic)
    scenario_results = {}
    for sc in ['Normal', 'Kompetitif', 'Booming']:
        sc_input = dict(input_data)
        sc_input['market_scenario'] = sc
        scenario_results[sc] = deterministic_predict(sc_input, df_skill_master)

# Store for Rencana Aksi page
st.session_state['prediction_result'] = result
st.session_state['scenario_results'] = scenario_results

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("# Hasil Simulasi")
st.markdown(f"**{input_data['target_role']}** | Latar Belakang: {input_data['background_level']} | "
            f"Skenario: {input_data['market_scenario']} | {input_data['study_hours_per_week']} jam/minggu")
if not model_loaded:
    st.caption("Menggunakan formula deterministik (model AI belum di-train)")
st.markdown("---")

# ============================================================
# KPI 1: SKOR KESENJANGAN KEAHLIAN (Skill Gap Score)
# ============================================================
st.markdown("### Skill Gap Score")

col_gauge, col_info = st.columns([2, 1])

# Mapping label kesiapan bahasa Indonesia untuk status gabungan
readiness_mapping = {
    "Ready": "Ready (Siap Kerja)",
    "Almost Ready": "Almost Ready (Hampir Siap Kerja)",
    "Needs Work": "Needs Work (Butuh Peningkatan)",
    "Significant Gap": "Significant Gap (Kesenjangan Signifikan)",
    "Major Gap": "Major Gap (Kesenjangan Besar)"
}

with col_gauge:
    gap = result['gap_score']
    
    # Premium color based on gap
    if gap <= 0.2: bar_color = "#0d9488"
    elif gap <= 0.4: bar_color = "#0f766e"
    elif gap <= 0.6: bar_color = "#f59e0b"
    elif gap <= 0.8: bar_color = "#e11d48"
    else: bar_color = "#9f1239"
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gap * 100,
        title={'text': "Skill Gap Score", 'font': {'size': 20, 'family': 'Inter'}},
        number={'suffix': '%', 'font': {'size': 40, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'bgcolor': '#1e293b',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 20], 'color': 'rgba(13, 148, 136, 0.2)'},     # Teal light opacity
                {'range': [20, 40], 'color': 'rgba(13, 148, 136, 0.4)'},     # Teal medium opacity
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.3)'},     # Amber light opacity
                {'range': [60, 80], 'color': 'rgba(225, 29, 72, 0.3)'},      # Rose light opacity
                {'range': [80, 100], 'color': 'rgba(159, 18, 57, 0.4)'},     # Rose deep opacity
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 3},
                'thickness': 0.75,
                'value': gap * 100,
            }
        }
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(t=50, b=20, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_info:
    readiness = result['readiness_label']
    readiness_display = readiness_mapping.get(readiness, readiness)
    
    st.markdown(f"## {readiness_display}")
    st.markdown(f"**Gap Score: {gap:.4f}**")
    st.markdown(f"""
    | Gap Range | Readiness Level |
    |---|---|
    | 0.0 - 0.2 | Ready (Siap Kerja) |
    | 0.2 - 0.4 | Almost Ready (Hampir Siap) |
    | 0.4 - 0.6 | Needs Work (Butuh Peningkatan) |
    | 0.6 - 0.8 | Significant Gap (Kesenjangan) |
    | 0.8 - 1.0 | Major Gap (Kesenjangan Besar) |
    """)

st.markdown("---")

# ============================================================
# KPI 2: TOP 3 MISSING SKILLS
# ============================================================
st.markdown("### Top 3 Missing Skills")

top3 = result['top_missing_skills']
cols_skills = st.columns(3)

for i, (col, skill) in enumerate(zip(cols_skills, top3)):
    with col:
        rank_label = ["Priority 1", "Priority 2", "Priority 3"][i]
        severity_color = "#e11d48" if skill['gap'] > 0.5 else "#f59e0b" if skill['gap'] > 0.3 else "#0d9488"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border-radius: 12px; padding: 1.5rem; text-align: center;
                    border-top: 4px solid {severity_color};
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
                    color: #f8fafc;">
            <h4 style="margin: 0; color: #94a3b8; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;">{rank_label}</h4>
            <h3 style="margin: 0.5rem 0; color: #f8fafc; font-weight: 700; font-size: 1.25rem;">{skill['skill_name']}</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">Category: {skill['category']}</p>
            <p style="margin: 0.25rem 0; font-size: 0.95rem; color: #cbd5e1;">Proficiency: <strong>{skill['proficiency']}/100</strong></p>
            <p style="margin: 0.25rem 0; font-size: 0.95rem; color: #cbd5e1;">Importance: <strong>{skill['importance']:.2f}</strong></p>
            <p style="margin: 0.25rem 0; font-size: 0.95rem; color: #cbd5e1;">Gap Score: <strong style="color: {severity_color};">{skill['gap']:.4f}</strong></p>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">~{skill['avg_learning_hours']} jam belajar</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# KPI 3: PREDIKSI WAKTU SIAP KERJA (Estimated Weeks to Job-Ready)
# ============================================================
st.markdown("### Estimated Time to Job-Ready")

col_time1, col_time2 = st.columns([1, 2])

with col_time1:
    weeks = result['estimated_weeks']
    months = weeks / 4.33
    
    st.metric("Estimated Time", f"{weeks:.0f} minggu", f"~{months:.1f} bulan")
    st.metric("Study Hours/Week", f"{input_data['study_hours_per_week']} jam")
    st.metric("Market Scenario", input_data['market_scenario'])

with col_time2:
    # Bar chart: hours needed per skill (top 10)
    all_gaps = result['all_skill_gaps']
    top10 = sorted(all_gaps, key=lambda x: x['gap'], reverse=True)[:10]
    
    fig_hours = go.Figure()
    fig_hours.add_trace(go.Bar(
        y=[s['skill_name'] for s in reversed(top10)],
        x=[s['gap'] * s['avg_learning_hours'] for s in reversed(top10)],
        orientation='h',
        marker_color=[
            '#e11d48' if s['gap'] > 0.5 else '#f59e0b' if s['gap'] > 0.3 else '#0d9488'
            for s in reversed(top10)
        ],
        text=[f"{s['gap'] * s['avg_learning_hours']:.0f} jam" for s in reversed(top10)],
        textposition='outside',
    ))
    fig_hours.update_layout(
        template="plotly_dark",
        title="Estimated Learning Hours per Skill (Top 10 Gaps)",
        xaxis_title="Learning Hours Required",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_hours, use_container_width=True)

st.markdown("---")

# ============================================================
# KPI 4: PERBANDINGAN SKENARIO KARIER (Career Scenario Comparison)
# ============================================================
st.markdown("### Career Scenario Comparison")

scenarios = ['Normal', 'Kompetitif', 'Booming']
sc_colors = ['#64748b', '#e11d48', '#0d9488']

col_sc1, col_sc2 = st.columns(2)

with col_sc1:
    # Gap Score comparison
    fig_gap_comp = go.Figure()
    for sc, color in zip(scenarios, sc_colors):
        fig_gap_comp.add_trace(go.Bar(
            name=sc,
            x=[sc],
            y=[scenario_results[sc]['gap_score']],
            marker_color=color,
            text=[f"{scenario_results[sc]['gap_score']:.4f}"],
            textposition='outside',
        ))
    fig_gap_comp.update_layout(
        template="plotly_dark",
        title="Gap Score per Scenario",
        yaxis_title="Gap Score",
        yaxis_range=[0, 1],
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_gap_comp, use_container_width=True)

with col_sc2:
    # Estimated weeks comparison
    fig_weeks_comp = go.Figure()
    for sc, color in zip(scenarios, sc_colors):
        fig_weeks_comp.add_trace(go.Bar(
            name=sc,
            x=[sc],
            y=[scenario_results[sc]['estimated_weeks']],
            marker_color=color,
            text=[f"{scenario_results[sc]['estimated_weeks']:.0f} minggu"],
            textposition='outside',
        ))
    fig_weeks_comp.update_layout(
        template="plotly_dark",
        title="Estimated Weeks per Scenario",
        yaxis_title="Estimated Weeks",
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_weeks_comp, use_container_width=True)

# Scenario comparison table
st.markdown("#### Scenario Comparison Summary")
sc_data = []
for sc in scenarios:
    r = scenario_results[sc]
    sc_data.append({
        'Scenario': sc,
        'Gap Score': f"{r['gap_score']:.4f}",
        'Readiness': r['readiness_label'],
        'Est. Weeks': f"{r['estimated_weeks']:.0f}",
        'Top Missing #1': r['top_missing_skills'][0]['skill_name'] if r['top_missing_skills'] else '-',
        'Top Missing #2': r['top_missing_skills'][1]['skill_name'] if len(r['top_missing_skills']) > 1 else '-',
        'Top Missing #3': r['top_missing_skills'][2]['skill_name'] if len(r['top_missing_skills']) > 2 else '-',
    })
st.dataframe(pd.DataFrame(sc_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# FULL SKILL BREAKDOWN
# ============================================================
with st.expander("Lihat Rincian Seluruh Keahlian (Full Skill Breakdown)"):
    all_gaps = result['all_skill_gaps']
    
    fig_full = go.Figure()
    sorted_skills = sorted(all_gaps, key=lambda x: x['importance'], reverse=False)
    
    fig_full.add_trace(go.Bar(
        y=[s['skill_name'] for s in sorted_skills],
        x=[s['importance'] * 100 for s in sorted_skills],
        name='Required Level',
        orientation='h',
        marker_color='rgba(225, 29, 72, 0.3)', # Rose light
    ))
    fig_full.add_trace(go.Bar(
        y=[s['skill_name'] for s in sorted_skills],
        x=[s['proficiency'] for s in sorted_skills],
        name='Your Proficiency',
        orientation='h',
        marker_color='rgba(13, 148, 136, 0.8)', # Teal solid
    ))
    
    fig_full.update_layout(
        template="plotly_dark",
        title="Your Proficiency vs Required Level",
        xaxis_title="Proficiency (0-100)",
        barmode='overlay',
        height=600,
        margin=dict(l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_full, use_container_width=True)

st.caption("Navigasi ke halaman Rencana Aksi untuk mendapatkan rencana belajar personal.")
