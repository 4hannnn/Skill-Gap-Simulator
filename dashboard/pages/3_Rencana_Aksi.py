"""
3_Rencana_Aksi.py
=================
Halaman Rencana Aksi — Action plan personal berbasis AI.
Menampilkan roadmap belajar, timeline, milestones, dan resources.
"""

import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJECT_ROOT)

from scripts.model.learning_strategy import generate_learning_strategy, RESOURCE_MAP

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Rencana Aksi | Simulator Kesenjangan Keahlian", page_icon=None, layout="wide")

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
# CHECK DATA
# ============================================================
if not st.session_state.get('prediction_result'):
    st.warning("Belum ada hasil simulasi. Silakan isi profil di halaman Input dan jalankan simulasi terlebih dahulu.")
    st.page_link("pages/1_Input.py", label="Buka Halaman Input", icon=None)
    st.stop()

result = st.session_state['prediction_result']
input_data = result.get('input_summary', st.session_state.get('input_data', {}))

# Generate strategy
strategy = generate_learning_strategy(result)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("# Rencana Aksi Personal")
st.markdown(f"**{input_data['target_role']}** | Latar Belakang: {input_data.get('background_level', 'N/A')} | "
            f"Skenario: {input_data.get('market_scenario', 'N/A')}")
st.markdown("---")

# Mapping label kesiapan bahasa Indonesia untuk status gabungan
readiness_mapping = {
    "Ready": "Ready (Siap Kerja)",
    "Almost Ready": "Almost Ready (Hampir Siap)",
    "Needs Work": "Needs Work (Butuh Peningkatan)",
    "Significant Gap": "Significant Gap (Kesenjangan)",
    "Major Gap": "Major Gap (Kesenjangan Besar)"
}

# ============================================================
# SUMMARY METRICS
# ============================================================
summary = strategy['summary']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Readiness Status", readiness_mapping.get(summary['current_status'], summary['current_status']))
col2.metric("Total Estimated Time", f"{summary['total_estimated_weeks']} minggu")
col3.metric("Skills to Improve", f"{summary['total_skills_to_improve']} keahlian")
col4.metric("Critical Skills", f"{summary['critical_skills']} keahlian")

st.markdown("---")

# ============================================================
# TIMELINE VISUALIZATION
# ============================================================
st.markdown("### Learning Roadmap")

milestones = strategy['milestones']
phases = strategy['phases']

# Gantt-like chart
fig_timeline = go.Figure()

colors_phase = ['#e11d48', '#f59e0b', '#0d9488'] # Premium Rose, Amber, Teal
start_week = 0

for i, phase in enumerate(phases):
    fig_timeline.add_trace(go.Bar(
        y=[f"Phase {phase['phase']}"],
        x=[phase['estimated_weeks']],
        base=[start_week],
        orientation='h',
        name=phase['name'],
        marker_color=colors_phase[i],
        text=[f"{phase['name']} ({phase['estimated_weeks']} mgg)"],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate=f"<b>{phase['name']}</b><br>"
                      f"Minggu {start_week+1} - {start_week + phase['estimated_weeks']}<br>"
                      f"Durasi: {phase['estimated_weeks']} minggu<extra></extra>",
    ))
    start_week += phase['estimated_weeks']

# Add milestone markers
for m in milestones:
    fig_timeline.add_vline(
        x=m['week'], line_dash="dash", line_color="rgba(248, 250, 252, 0.4)",
        annotation_text=f"{m['milestone'][:25]}",
        annotation_position="top",
    )

fig_timeline.update_layout(
    template="plotly_dark",
    title="Learning Roadmap",
    xaxis_title="Weeks",
    barmode='stack',
    height=200,
    showlegend=False,
    margin=dict(l=10, r=10, t=50, b=30),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter")
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")

# ============================================================
# PHASE DETAILS
# ============================================================
st.markdown("### Detail per Fase (Phase Details)")

for phase in phases:
    with st.expander(f"{phase['name']} — {phase['estimated_weeks']} minggu", expanded=(phase['phase'] == 1)):
        st.markdown(f"*{phase['description']}*")
        
        if not phase['skills']:
            st.info("Tidak ada keahlian yang perlu ditingkatkan di fase ini!")
            continue
        
        for skill in phase['skills']:
            col_skill, col_progress, col_resource = st.columns([2, 2, 2])
            
            with col_skill:
                severity_label = {
                    'Critical': 'Critical (Kritis)',
                    'High': 'High (Tinggi)',
                    'Medium': 'Medium (Sedang)',
                    'Low': 'Low (Rendah)',
                    'Kritis': 'Critical (Kritis)',
                    'Tinggi': 'High (Tinggi)',
                    'Sedang': 'Medium (Sedang)',
                    'Rendah': 'Low (Rendah)'
                }.get(skill['gap_severity'], skill['gap_severity'])
                
                st.markdown(f"**{skill['skill_name']}** ({skill['category']})")
                st.caption(f"Priority: {severity_label} | ~{skill['hours_needed']} jam belajar")
            
            with col_progress:
                # Progress bar
                progress_pct = skill['current_proficiency'] / max(skill['target_proficiency'], 1)
                st.progress(min(1.0, progress_pct))
                st.caption(f"Kemahiran saat ini: {skill['current_proficiency']} → Target: {skill['target_proficiency']}/100")
            
            with col_resource:
                st.markdown("**Referensi Belajar:**")
                for res in skill['resources']:
                    st.markdown(f"- {res}")
            
            st.markdown("---")

# ============================================================
# WEEKLY PLAN
# ============================================================
st.markdown("### Weekly Plan (First 4 Weeks)")

weekly_plan = strategy['weekly_plan']

if weekly_plan:
    for week in weekly_plan:
        with st.container():
            st.markdown(f"#### Week {week['week']} ({week['hours']} hours)")
            
            cols_act = st.columns(len(week['activities']))
            for j, (col, activity) in enumerate(zip(cols_act, week['activities'])):
                with col:
                    # Remove emoji, style with premium Slate/Teal colors for dark mode compatibility
                    st.markdown(f"""
                    <div style="background: #1e293b; border-radius: 8px; 
                                padding: 1rem; border-left: 3px solid #0d9488;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                        <p style="margin: 0; color: #f8fafc; font-weight: 500;">{activity}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("")
else:
    st.info("Anda sudah siap! Tidak perlu rencana belajar tambahan.")

st.markdown("---")

# ============================================================
# MILESTONES
# ============================================================
st.markdown("### Target Pencapaian (Milestones)")

cols_ms = st.columns(len(milestones))
for col, m in zip(cols_ms, milestones):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
                    border-radius: 12px; padding: 1.5rem; text-align: center; color: white;
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; color: white; font-weight: 800; font-size: 1.5rem;">Week {m['week']}</h2>
            <p style="font-size: 1.1rem; margin: 0.5rem 0; font-weight: 500;">{m['milestone']}</p>
            <p style="opacity: 0.9; margin: 0; font-size: 0.9rem;">Expected Gap: {m['expected_gap_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SKILL PRIORITY MATRIX
# ============================================================
st.markdown("### Skill Priority Matrix")

all_gaps = result['all_skill_gaps']
if all_gaps:
    df_gaps_plot = pd.DataFrame(all_gaps)
    
    fig_matrix = px.scatter(
        df_gaps_plot,
        x='proficiency',
        y='importance',
        size='gap',
        color='category',
        hover_name='skill_name',
        title="Importance vs Proficiency (Bubble Size = Skill Gap)",
        labels={
            'proficiency': 'Your Proficiency (0-100)',
            'importance': 'Importance Score (0-1)',
            'gap': 'Gap Score',
            'category': 'Category',
        },
        color_discrete_map={
            'Technical': '#0d9488',     # Teal
            'Tool': '#64748b',          # Slate
            'Soft Skill': '#e11d48',     # Rose
        },
        height=450,
    )
    
    # Add quadrant lines
    fig_matrix.add_hline(y=0.75, line_dash="dash", line_color="#94a3b8", opacity=0.7,
                         annotation_text="High Importance", annotation_position="top left")
    fig_matrix.add_vline(x=50, line_dash="dash", line_color="#94a3b8", opacity=0.7,
                         annotation_text="Medium Proficiency", annotation_position="top right")
    
    # Quadrant labels with premium styling and colors
    fig_matrix.add_annotation(x=25, y=0.9, text="CRITICAL PRIORITY (Prioritas Utama)",
                             showarrow=False, font=dict(size=12, color="#e11d48", family='Inter'))
    fig_matrix.add_annotation(x=75, y=0.9, text="KEEP UP (Sudah Baik)",
                             showarrow=False, font=dict(size=12, color="#0d9488", family='Inter'))
    fig_matrix.add_annotation(x=25, y=0.3, text="SECONDARY PRIORITY (Perkuat Nanti)",
                             showarrow=False, font=dict(size=12, color="#d97706", family='Inter'))
    fig_matrix.add_annotation(x=75, y=0.3, text="NICE TO HAVE (Keahlian Pendukung)",
                             showarrow=False, font=dict(size=12, color="#64748b", family='Inter'))
    
    fig_matrix.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

st.markdown("---")
st.caption("Tips: Fokus pada keahlian di kuadran kiri atas (CRITICAL PRIORITY / Prioritas Utama) — kepentingan tinggi namun kemahiran Anda masih rendah.")
st.caption("Capstone Project CC26-PRU469 | Coding Camp 2026 didukung oleh DBS Foundation")
