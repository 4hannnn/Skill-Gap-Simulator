"""
generate_user_profiles.py
=========================
Script untuk menghasilkan 750 user profiles sintetis
dengan skill proficiency yang bervariasi secara realistis.

Output: data/raw/user_profiles_raw.csv
"""

import pandas as pd
import numpy as np
import random
import json
import os

# Set seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# Import skill taxonomy dari generate_job_postings
import sys
sys.path.insert(0, os.path.dirname(__file__))
from generate_job_postings import SKILL_TAXONOMY

# ============================================================
# KONFIGURASI USER PROFILES
# ============================================================

N_USERS_PER_ROLE = 1500
ROLES = ["Data Analyst", "Frontend Developer", "UI/UX Designer"]
MARKET_SCENARIOS = ["Normal", "Kompetitif", "Booming"]
BACKGROUND_LEVELS = ["Pemula", "Menengah", "Lanjutan"]

# Background level → proficiency range mapping
PROFICIENCY_RANGES = {
    "Pemula": {"mean": 25, "std": 15, "min": 0, "max": 50},
    "Menengah": {"mean": 55, "std": 15, "min": 20, "max": 80},
    "Lanjutan": {"mean": 78, "std": 12, "min": 50, "max": 100},
}

# Background level distribution
BACKGROUND_WEIGHTS = [0.40, 0.35, 0.25]  # Lebih banyak pemula (realistis)

# Study hours distribution per background
STUDY_HOURS_RANGES = {
    "Pemula": {"mean": 12, "std": 5, "min": 3, "max": 25},
    "Menengah": {"mean": 10, "std": 4, "min": 3, "max": 20},
    "Lanjutan": {"mean": 7, "std": 3, "min": 3, "max": 15},
}

# ============================================================
# DIRTY DATA CONFIG (moderate)
# ============================================================

DIRT_CONFIG = {
    "missing_rate": 0.08,
    "duplicate_rate": 0.03,
    "near_duplicate_rate": 0.015,
    "typo_rate": 0.08,
    "outlier_rate": 0.04,
}

BACKGROUND_VARIANTS = [
    "pemula", "Beginner", "Menengah ", "Lanjut", "n/a",
]

SCENARIO_VARIANTS = [
    "normal", "Kompetitif ", "Booming", "competitive", "n/a",
]

SKILL_TYPO_MAP = {
    "JavaScript": ["Java Script", "Javasript", "JS", "ReactJS"],
    "Python": ["Pyhton", "python ", "PYTHON"],
    "SQL": ["Sql", "S Q L"],
    "Power BI": ["PowerBI", "Power-BI", "poweR bi"],
    "Excel": ["Ms Excel", "MS-Excel", "Excell"],
}


def _random_mask(n, rate):
    return np.random.rand(n) < rate


def _dirty_text(value):
    if not isinstance(value, str) or value == "":
        return value
    text = value
    if random.random() < 0.4:
        text = text.lower()
    if random.random() < 0.3:
        text = text.upper()
    if random.random() < 0.4:
        text = f" {text} "
    return text


def get_role_skills(role):
    """Ambil semua skills untuk sebuah role."""
    skills = []
    for cat_skills in SKILL_TAXONOMY[role].values():
        skills.extend(cat_skills)
    return skills


def generate_user_proficiency(role, background_level):
    """
    Generate proficiency untuk setiap skill berdasarkan background level.
    Proficiency bervariasi per skill — bukan flat sama semua.
    """
    skills = get_role_skills(role)
    prof_range = PROFICIENCY_RANGES[background_level]
    
    proficiency = {}
    for skill_info in skills:
        # Base proficiency dari background level
        base = np.random.normal(prof_range["mean"], prof_range["std"])
        
        # Skill kategori Soft Skill cenderung lebih tinggi (orang biasanya lebih percaya diri)
        if skill_info["category"] == "Soft Skill":
            base += np.random.uniform(5, 15)
        
        # Tool skills — bisa sangat bervariasi
        if skill_info["category"] == "Tool":
            base += np.random.uniform(-10, 10)
        
        # Clip ke range valid
        base = np.clip(base, prof_range["min"], prof_range["max"])
        proficiency[skill_info["skill"]] = round(base)
    
    return proficiency


def compute_gap_score(proficiency, role, scenario="Normal"):
    """
    Hitung Weighted Gap Score:
    Gap Score = 1 - (Σ (user_skill_i × importance_i) / Σ importance_i)
    """
    skills = get_role_skills(role)
    
    # Market demand column mapping
    demand_map = {
        "Normal": "importance_score",
        "Kompetitif": "importance_score",
        "Booming": "importance_score",
    }
    
    # Market multiplier
    multiplier_map = {
        "Normal": 1.0,
        "Kompetitif": 1.15,
        "Booming": 0.90,
    }
    
    multiplier = multiplier_map[scenario]
    
    weighted_sum = 0
    importance_sum = 0
    
    for skill_info in skills:
        skill_name = skill_info["skill"]
        importance = min(1.0, skill_info["importance"] * multiplier)
        user_level = proficiency.get(skill_name, 0) / 100.0  # Normalize 0-1
        
        weighted_sum += user_level * importance
        importance_sum += importance
    
    if importance_sum == 0:
        return 1.0
    
    gap_score = 1 - (weighted_sum / importance_sum)
    return round(gap_score, 4)


def find_top_missing_skills(proficiency, role, n=3):
    """Temukan top-N skills dengan gap terbesar (importance tinggi, proficiency rendah)."""
    skills = get_role_skills(role)
    
    gaps = []
    for skill_info in skills:
        skill_name = skill_info["skill"]
        user_level = proficiency.get(skill_name, 0) / 100.0
        importance = skill_info["importance"]
        
        # Gap = importance × (1 - proficiency)
        gap = importance * (1 - user_level)
        gaps.append((skill_name, round(gap, 4)))
    
    # Sort by gap descending
    gaps.sort(key=lambda x: x[1], reverse=True)
    return [g[0] for g in gaps[:n]]


def estimate_weeks_ready(proficiency, role, study_hours_per_week, scenario="Normal"):
    """
    Estimasi minggu siap kerja:
    Weeks = Σ (skill_gap_i × avg_learning_hours_i × importance_i) / study_hours_per_week
    × market_multiplier
    """
    skills = get_role_skills(role)
    
    multiplier_map = {
        "Normal": 1.0,
        "Kompetitif": 1.3,
        "Booming": 0.8,
    }
    multiplier = multiplier_map[scenario]
    
    total_hours_needed = 0
    for skill_info in skills:
        skill_name = skill_info["skill"]
        user_level = proficiency.get(skill_name, 0) / 100.0
        importance = skill_info["importance"]
        avg_hours = skill_info["avg_learning_hours"]
        
        # Hours needed = gap × avg_learning_hours × importance
        skill_gap = max(0, 1 - user_level)
        hours_needed = skill_gap * avg_hours * importance
        total_hours_needed += hours_needed
    
    if study_hours_per_week <= 0:
        return 999
    
    weeks = (total_hours_needed / study_hours_per_week) * multiplier
    return round(weeks, 1)


def get_readiness_label(gap_score):
    """Konversi gap score ke label kesiapan."""
    if gap_score <= 0.2:
        return "Ready"
    elif gap_score <= 0.4:
        return "Almost Ready"
    elif gap_score <= 0.6:
        return "Needs Work"
    elif gap_score <= 0.8:
        return "Significant Gap"
    else:
        return "Major Gap"


def main():
    print("=" * 60)
    print("GENERATE USER PROFILES (750 Users)")
    print("=" * 60)
    
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    all_users = []
    user_skills_records = []
    user_id = 1
    
    for role in ROLES:
        print(f"\n  Generating {N_USERS_PER_ROLE} users untuk {role}...")
        
        for i in range(N_USERS_PER_ROLE):
            # Random background & scenario
            background = np.random.choice(BACKGROUND_LEVELS, p=BACKGROUND_WEIGHTS)
            scenario = random.choice(MARKET_SCENARIOS)
            
            # Study hours
            sh_range = STUDY_HOURS_RANGES[background]
            study_hours = round(np.clip(
                np.random.normal(sh_range["mean"], sh_range["std"]),
                sh_range["min"], sh_range["max"]
            ), 1)
            
            # Generate proficiency
            proficiency = generate_user_proficiency(role, background)
            
            # Compute metrics
            gap_score = compute_gap_score(proficiency, role, scenario)
            top_missing = find_top_missing_skills(proficiency, role, n=3)
            est_weeks = estimate_weeks_ready(proficiency, role, study_hours, scenario)
            readiness = get_readiness_label(gap_score)
            
            # User record
            user_record = {
                "user_id": f"USR-{user_id:04d}",
                "target_role": role,
                "background_level": background,
                "study_hours_per_week": study_hours,
                "market_scenario": scenario,
                "gap_score": gap_score,
                "readiness_label": readiness,
                "top_missing_skill_1": top_missing[0],
                "top_missing_skill_2": top_missing[1],
                "top_missing_skill_3": top_missing[2],
                "estimated_weeks_ready": est_weeks,
                "current_skills_json": json.dumps(proficiency),
            }
            all_users.append(user_record)
            
            # Per-skill records (untuk analisis detail)
            for skill_name, level in proficiency.items():
                user_skills_records.append({
                    "user_id": f"USR-{user_id:04d}",
                    "target_role": role,
                    "skill_name": skill_name,
                    "proficiency_level": level,
                    "background_level": background,
                    "market_scenario": scenario,
                })
            
            user_id += 1
    
    # Save
    df_users = pd.DataFrame(all_users)
    df_user_skills = pd.DataFrame(user_skills_records)

    # ========================================================
    # DIRTY DATA INJECTION (moderate)
    # ========================================================
    n_users = len(df_users)
    n_skills = len(df_user_skills)

    # Missing values in user profiles
    for col in ["background_level", "market_scenario", "study_hours_per_week"]:
        mask = _random_mask(n_users, DIRT_CONFIG["missing_rate"])
        df_users.loc[mask, col] = None

    # Categorical variants
    mask = _random_mask(n_users, DIRT_CONFIG["typo_rate"])
    df_users.loc[mask, "background_level"] = np.random.choice(
        BACKGROUND_VARIANTS, size=mask.sum()
    )
    mask = _random_mask(n_users, DIRT_CONFIG["typo_rate"])
    df_users.loc[mask, "market_scenario"] = np.random.choice(
        SCENARIO_VARIANTS, size=mask.sum()
    )

    # Outlier study hours
    mask = _random_mask(n_users, DIRT_CONFIG["outlier_rate"])
    df_users.loc[mask, "study_hours_per_week"] = np.random.choice(
        [-5, 0, 60, 80], size=mask.sum()
    )

    # Corrupt some computed fields
    mask = _random_mask(n_users, DIRT_CONFIG["missing_rate"] / 2)
    df_users.loc[mask, "gap_score"] = None
    mask = _random_mask(n_users, DIRT_CONFIG["missing_rate"] / 2)
    df_users.loc[mask, "readiness_label"] = "unknown"
    mask = _random_mask(n_users, DIRT_CONFIG["missing_rate"] / 2)
    df_users.loc[mask, "estimated_weeks_ready"] = None

    # Corrupt some JSON blobs
    mask = _random_mask(n_users, DIRT_CONFIG["typo_rate"])
    df_users.loc[mask, "current_skills_json"] = "{bad_json"

    # Dirty user skills
    mask = _random_mask(n_skills, DIRT_CONFIG["typo_rate"])
    for idx in df_user_skills[mask].index:
        skill = df_user_skills.at[idx, "skill_name"]
        if skill in SKILL_TYPO_MAP:
            df_user_skills.at[idx, "skill_name"] = random.choice(SKILL_TYPO_MAP[skill])
        else:
            df_user_skills.at[idx, "skill_name"] = _dirty_text(skill)

    # Outlier proficiency levels
    mask = _random_mask(n_skills, DIRT_CONFIG["outlier_rate"])
    df_user_skills.loc[mask, "proficiency_level"] = np.random.choice(
        [-10, 120, 150], size=mask.sum()
    )

    # Missing identifiers
    mask = _random_mask(n_skills, DIRT_CONFIG["missing_rate"] / 2)
    df_user_skills.loc[mask, "user_id"] = None
    mask = _random_mask(n_skills, DIRT_CONFIG["missing_rate"] / 2)
    df_user_skills.loc[mask, "skill_name"] = None

    # Duplicates
    dup_count = max(1, int(n_users * DIRT_CONFIG["duplicate_rate"]))
    df_users = pd.concat([df_users, df_users.sample(dup_count, random_state=42)], ignore_index=True)
    dup_count = max(1, int(n_skills * DIRT_CONFIG["duplicate_rate"]))
    df_user_skills = pd.concat([
        df_user_skills, df_user_skills.sample(dup_count, random_state=99)
    ], ignore_index=True)
    
    users_path = os.path.join(raw_dir, "user_profiles_raw.csv")
    user_skills_path = os.path.join(raw_dir, "user_skills_raw.csv")
    
    df_users.to_csv(users_path, index=False)
    df_user_skills.to_csv(user_skills_path, index=False)
    
    # Summary
    print(f"\n  [OK] User profiles: {len(df_users)} rows -> {users_path}")
    print(f"  [OK] User skills detail: {len(df_user_skills)} rows -> {user_skills_path}")
    
    print("\n  Summary Statistics:")
    print(f"  - Total users: {len(df_users)}")
    for role in ROLES:
        subset = df_users[df_users["target_role"] == role]
        print(f"  - {role}:")
        print(f"    - Count: {len(subset)}")
        print(f"    - Avg gap score: {subset['gap_score'].mean():.3f}")
        print(f"    - Avg est. weeks: {subset['estimated_weeks_ready'].mean():.1f}")
        print(f"    - Background dist: {subset['background_level'].value_counts().to_dict()}")
    
    print("\n[SUCCESS] Semua user profiles berhasil di-generate!")
    print("=" * 60)


if __name__ == "__main__":
    main()
