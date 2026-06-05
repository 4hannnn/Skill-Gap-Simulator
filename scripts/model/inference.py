"""
inference.py
============
Script inferensi untuk memprediksi skill gap user baru
menggunakan model yang sudah di-train.

Usage:
    python scripts/model/inference.py

Atau import sebagai module:
    from scripts.model.inference import predict_user
"""

import os
import sys
import json
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from train_model import WeightedGapLoss

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(BASE_DIR, "models", "skill_gap_model")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_model_and_metadata():
    """Load saved model dan metadata."""
    
    model_path = os.path.join(MODEL_DIR, "skill_gap_model.keras")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load model with custom objects
    model = keras.models.load_model(
        model_path,
        custom_objects={'WeightedGapLoss': WeightedGapLoss}
    )
    
    # Load skill master for analysis
    df_skill_master = pd.read_csv(os.path.join(PROCESSED_DIR, "skill_master_cleaned.csv"))
    
    return model, metadata, df_skill_master


def prepare_input(target_role, background_level, market_scenario,
                  study_hours_per_week, skill_proficiencies, metadata):
    """
    Prepare input features untuk inferensi.
    
    Parameters:
    -----------
    target_role : str
        "Data Analyst", "Frontend Developer", atau "UI/UX Designer"
    background_level : str
        "Pemula", "Menengah", atau "Lanjutan"
    market_scenario : str
        "Normal", "Kompetitif", atau "Booming"
    study_hours_per_week : float
        Jam belajar per minggu
    skill_proficiencies : dict
        {skill_name: proficiency_level (0-100)}
    metadata : dict
        Model metadata
    
    Returns:
    --------
    np.array: Input features array
    """
    
    # One-hot: role
    role_features = [1 if f"role_{target_role}" == c else 0 
                     for c in metadata['role_classes']]
    
    # One-hot: background
    bg_features = [1 if f"bg_{background_level}" == c else 0 
                   for c in metadata['bg_classes']]
    
    # One-hot: scenario
    scenario_features = [1 if f"scenario_{market_scenario}" == c else 0 
                         for c in metadata['scenario_classes']]
    
    # Scale study hours
    sh_params = metadata['study_hours_params']
    study_hours_scaled = (study_hours_per_week - sh_params['mean']) / sh_params['scale']
    
    # Skill features (order must match training)
    skill_cols = metadata['skill_columns']
    skill_features = []
    for col in skill_cols:
        # Proficiency 0-100 → 0-1
        prof = skill_proficiencies.get(col, 0) / 100.0
        skill_features.append(prof)
    
    # Combine
    features = (role_features + bg_features + scenario_features + 
                [study_hours_scaled] + skill_features)
    
    return np.array([features], dtype=np.float32)


def predict_user(target_role, background_level, market_scenario,
                 study_hours_per_week, skill_proficiencies,
                 model=None, metadata=None, df_skill_master=None):
    """
    Prediksi skill gap untuk satu user.
    
    Returns:
    --------
    dict with keys:
        - gap_score: float (0-1)
        - readiness_label: str
        - readiness_probabilities: dict
        - estimated_weeks: float
        - top_missing_skills: list of dict
        - scenario_comparison: dict (jika diminta)
    """
    
    # Load model if not provided
    if model is None:
        model, metadata, df_skill_master = load_model_and_metadata()
    
    # Prepare input
    X = prepare_input(target_role, background_level, market_scenario,
                      study_hours_per_week, skill_proficiencies, metadata)
    
    # Predict
    pred_gap, pred_readiness, pred_weeks = model.predict(X, verbose=0)
    
    # Decode outputs
    gap_score = float(pred_gap[0][0])
    
    readiness_labels = metadata['readiness_labels']
    readiness_probs = {label: float(prob) 
                       for label, prob in zip(readiness_labels, pred_readiness[0])}
    predicted_label = readiness_labels[np.argmax(pred_readiness[0])]
    
    # Decode weeks (inverse scale)
    weeks_params = metadata['weeks_params']
    estimated_weeks = float(pred_weeks[0][0] * weeks_params['scale'] + weeks_params['mean'])
    estimated_weeks = max(0, estimated_weeks)  # Clip negative
    
    # Find top missing skills
    role_skills = df_skill_master[df_skill_master['role'] == target_role]
    skill_gaps = []
    for _, row in role_skills.iterrows():
        user_level = skill_proficiencies.get(row['skill_name'], 0) / 100.0
        importance = row['importance_score']
        gap = importance * (1 - user_level)
        skill_gaps.append({
            'skill_name': row['skill_name'],
            'category': row['skill_category'],
            'proficiency': skill_proficiencies.get(row['skill_name'], 0),
            'importance': importance,
            'gap': round(gap, 4),
            'avg_learning_hours': int(row['avg_learning_hours']),
        })
    
    skill_gaps.sort(key=lambda x: x['gap'], reverse=True)
    
    result = {
        'gap_score': round(gap_score, 4),
        'readiness_label': predicted_label,
        'readiness_probabilities': readiness_probs,
        'estimated_weeks': round(estimated_weeks, 1),
        'top_missing_skills': skill_gaps[:3],
        'all_skill_gaps': skill_gaps,
        'input_summary': {
            'target_role': target_role,
            'background_level': background_level,
            'market_scenario': market_scenario,
            'study_hours_per_week': study_hours_per_week,
        }
    }
    
    return result


def predict_all_scenarios(target_role, background_level,
                          study_hours_per_week, skill_proficiencies,
                          model=None, metadata=None, df_skill_master=None):
    """Prediksi untuk semua 3 skenario pasar sekaligus."""
    
    if model is None:
        model, metadata, df_skill_master = load_model_and_metadata()
    
    results = {}
    for scenario in ['Normal', 'Kompetitif', 'Booming']:
        results[scenario] = predict_user(
            target_role, background_level, scenario,
            study_hours_per_week, skill_proficiencies,
            model, metadata, df_skill_master
        )
    
    return results


# ============================================================
# DEMO
# ============================================================
def demo():
    """Demo inferensi dengan sample user."""
    
    print("=" * 60)
    print("SKILL GAP SIMULATOR — INFERENCE DEMO")
    print("=" * 60)
    
    model, metadata, df_skill_master = load_model_and_metadata()
    
    # Sample user: Data Analyst pemula
    skill_proficiencies = {
        "Python": 30, "SQL": 25, "Statistics": 20,
        "Data Modeling": 15, "ETL Process": 10,
        "Hypothesis Testing": 10, "Regression Analysis": 5,
        "A/B Testing": 5, "Excel": 60, "Tableau": 20,
        "Power BI": 15, "Pandas": 25, "NumPy": 20,
        "Google Sheets": 50, "Jupyter Notebook": 30,
        "Analytical Thinking": 40, "Communication": 50,
        "Problem Solving": 35, "Attention to Detail": 45,
        "Business Acumen": 15,
    }
    
    # Configure stdout to use utf-8 encoding if possible to prevent Windows encoding crashes
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("\n[Sample User]")
    print(f"  Role: Data Analyst")
    print(f"  Background: Pemula")
    print(f"  Study Hours: 10 jam/minggu")
    
    # Single prediction
    result = predict_user(
        target_role="Data Analyst",
        background_level="Pemula",
        market_scenario="Normal",
        study_hours_per_week=10,
        skill_proficiencies=skill_proficiencies,
        model=model, metadata=metadata, df_skill_master=df_skill_master,
    )
    
    print(f"\n[Hasil Prediksi (Skenario Normal)]")
    print(f"  Gap Score: {result['gap_score']:.4f}")
    print(f"  Readiness: {result['readiness_label']}")
    print(f"  Est. Weeks: {result['estimated_weeks']:.1f} minggu")
    print(f"\n  Top 3 Missing Skills:")
    for i, skill in enumerate(result['top_missing_skills'], 1):
        print(f"    {i}. {skill['skill_name']} "
              f"(prof: {skill['proficiency']}, imp: {skill['importance']:.2f}, "
              f"gap: {skill['gap']:.4f})")
    
    # All scenarios
    print(f"\n[Career Scenario Comparison]")
    scenarios = predict_all_scenarios(
        "Data Analyst", "Pemula", 10, skill_proficiencies,
        model, metadata, df_skill_master,
    )
    
    print(f"  {'Skenario':12s} | {'Gap Score':>10s} | {'Readiness':>18s} | {'Weeks':>8s}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*18} | {'-'*8}")
    for sc, res in scenarios.items():
        print(f"  {sc:12s} | {res['gap_score']:10.4f} | {res['readiness_label']:>18s} | {res['estimated_weeks']:8.1f}")
    
    print(f"\n[Inference demo selesai!]")
    

if __name__ == "__main__":
    demo()
