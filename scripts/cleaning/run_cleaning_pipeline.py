"""
run_cleaning_pipeline.py
========================
Pipeline untuk membersihkan data sintetis "kotor" dan menghasilkan
file processed yang siap dipakai untuk modeling dan dashboard.

Input (raw):
- data/raw/job_postings_raw.csv
- data/raw/job_skills_raw.csv
- data/raw/skill_master_raw.csv
- data/raw/user_profiles_raw.csv
- data/raw/user_skills_raw.csv

Output (processed):
- data/processed/job_postings_cleaned.csv
- data/processed/job_skills_cleaned.csv
- data/processed/skill_master_cleaned.csv
- data/processed/user_profiles_cleaned.csv
- data/processed/user_skills_cleaned.csv
- data/processed/skill_frequency.csv
- data/processed/modeling_dataset.csv
"""

import os
import re
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

ROLE_MAP = {
    "data analyst": "Data Analyst",
    "frontend developer": "Frontend Developer",
    "ui/ux designer": "UI/UX Designer",
    "ux designer": "UI/UX Designer",
    "ui designer": "UI/UX Designer",
}

JOB_LEVEL_MAP = {
    "entry level": "Entry level",
    "entry-level": "Entry level",
    "associate": "Associate",
    "mid level": "Mid-Senior level",
    "mid-senior": "Mid-Senior level",
    "mid-senior level": "Mid-Senior level",
    "senior": "Mid-Senior level",
    "seniorr": "Mid-Senior level",
    "director": "Director",
    "executive": "Executive",
}

JOB_TYPE_MAP = {
    "full time": "Full-time",
    "full-time": "Full-time",
    "part time": "Part-time",
    "part-time": "Part-time",
    "contract": "Contract",
    "contractor": "Contract",
    "intern": "Internship",
    "internship": "Internship",
    "freelance": "Freelance",
}

BACKGROUND_MAP = {
    "pemula": "Pemula",
    "beginner": "Pemula",
    "menengah": "Menengah",
    "lanjut": "Lanjutan",
    "lanjutan": "Lanjutan",
}

SCENARIO_MAP = {
    "normal": "Normal",
    "kompetitif": "Kompetitif",
    "competitive": "Kompetitif",
    "booming": "Booming",
}

SKILL_CATEGORY_MAP = {
    "technical": "Technical",
    "tool": "Tool",
    "soft skill": "Soft Skill",
    "softskill": "Soft Skill",
}

SKILL_ALIAS_MAP = {
    "java script": "JavaScript",
    "javasript": "JavaScript",
    "js": "JavaScript",
    "reactjs": "React",
    "pyhton": "Python",
    "python": "Python",
    "sql": "SQL",
    "s q l": "SQL",
    "powerbi": "Power BI",
    "power-bi": "Power BI",
    "ms excel": "Excel",
    "ms-excel": "Excel",
    "excell": "Excel",
    "a/btesting": "A/B Testing",
    "ab testing": "A/B Testing",
    "machine-learning": "Machine Learning",
    "ml": "Machine Learning",
    "userresearch": "User Research",
    "user-research": "User Research",
    "designsystem": "Design Systems",
    "design-systems": "Design Systems",
}

ACRONYM_SKILLS = {"SQL", "HTML", "CSS", "ETL", "UI", "UX", "AI"}


def _normalize_text(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",")
    return text


def _smart_title(value):
    if not isinstance(value, str):
        return value
    text = _normalize_text(value)
    if text.isupper() and len(text) <= 5:
        return text
    text = text.title()
    text = text.replace("Ui/Ux", "UI/UX")
    return text


def _normalize_role(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return ROLE_MAP.get(key)


def _normalize_job_level(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return JOB_LEVEL_MAP.get(key)


def _normalize_job_type(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return JOB_TYPE_MAP.get(key)


def _normalize_background(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return BACKGROUND_MAP.get(key)


def _normalize_scenario(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return SCENARIO_MAP.get(key)


def _normalize_skill_category(value):
    if not isinstance(value, str):
        return None
    key = _normalize_text(value).lower()
    return SKILL_CATEGORY_MAP.get(key)


def _normalize_skill_name(value):
    if not isinstance(value, str):
        return None
    text = _normalize_text(value)
    key = text.lower().replace("-", " ").replace("/", " ")
    key = re.sub(r"\s+", " ", key)
    if key in SKILL_ALIAS_MAP:
        return SKILL_ALIAS_MAP[key]
    words = [w.upper() if w.upper() in ACRONYM_SKILLS else w.title() for w in key.split(" ")]
    return " ".join(words)


def _load_raw(name):
    path = os.path.join(RAW_DIR, name)
    return pd.read_csv(path)


def _save_processed(df, name):
    path = os.path.join(PROCESSED_DIR, name)
    df.to_csv(path, index=False)
    return path


def _build_skill_master(raw_master):
    df = raw_master.copy()

    df["role"] = df["role"].apply(_normalize_role)
    df["skill_name"] = df["skill_name"].apply(_normalize_skill_name)
    df["skill_category"] = df["skill_category"].apply(_normalize_skill_category)

    for col in ["importance_score", "market_demand_normal", "market_demand_competitive", "market_demand_booming"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)

    df["avg_learning_hours"] = pd.to_numeric(df["avg_learning_hours"], errors="coerce").clip(20, 400)

    df = df.dropna(subset=["role", "skill_name", "skill_category"])

    # Fill missing values by category/role medians
    df["importance_score"] = df.groupby(["role"])['importance_score'].transform(
        lambda s: s.fillna(s.median())
    )
    df["avg_learning_hours"] = df.groupby(["skill_category"])['avg_learning_hours'].transform(
        lambda s: s.fillna(s.median())
    )

    for col in ["market_demand_normal", "market_demand_competitive", "market_demand_booming"]:
        df[col] = df.groupby(["role"]) [col].transform(lambda s: s.fillna(s.median()))

    df = df.drop_duplicates(subset=["role", "skill_name"], keep="first")

    return df.reset_index(drop=True)


def _clean_job_postings(df_postings):
    df = df_postings.copy()

    df["job_title"] = df["job_title"].apply(_smart_title)
    df["company"] = df["company"].apply(_smart_title)
    df["job_location"] = df["job_location"].apply(_smart_title)
    df["role_category"] = df["role_category"].apply(_normalize_role)

    df["job_level"] = df["job_level"].apply(_normalize_job_level)
    df["job_type"] = df["job_type"].apply(_normalize_job_type)

    df["job_title"] = df["job_title"].fillna("Unknown Title")

    mode_level = df["job_level"].mode(dropna=True)
    mode_type = df["job_type"].mode(dropna=True)

    df["job_level"] = df["job_level"].fillna(mode_level.iloc[0] if not mode_level.empty else "Mid-Senior level")
    df["job_type"] = df["job_type"].fillna(mode_type.iloc[0] if not mode_type.empty else "Full-time")

    df["role_category"] = df["role_category"].fillna("Data Analyst")

    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    median_date = df["first_seen"].dropna().median()
    df["first_seen"] = df["first_seen"].fillna(median_date)

    # Remove duplicates by key fields
    df = df.drop_duplicates(subset=["job_id", "job_title", "company", "first_seen"])

    # Fix job_id missing or duplicate
    df["job_id"] = df["job_id"].fillna("")
    df["job_id"] = df["job_id"].astype(str).str.strip()
    existing_ids = df["job_id"].tolist()
    next_id = 1

    def _assign_id(current):
        nonlocal next_id
        if current and current != "nan":
            return current
        while True:
            new_id = f"JOB-{next_id:05d}"
            next_id += 1
            if new_id not in existing_ids:
                existing_ids.append(new_id)
                return new_id

    df["job_id"] = df["job_id"].apply(_assign_id)

    df["month"] = df["first_seen"].dt.month.astype(int)
    df["quarter"] = df["first_seen"].dt.quarter.astype(int)

    df["first_seen"] = df["first_seen"].dt.strftime("%Y-%m-%d")

    return df.reset_index(drop=True)


def _clean_job_skills(df_skills, df_postings, df_skill_master):
    df = df_skills.copy()

    df["job_id"] = df["job_id"].astype(str).str.strip()
    df["skill_name"] = df["skill_name"].apply(_normalize_skill_name)
    df["skill_category"] = df["skill_category"].apply(_normalize_skill_category)
    df["role_category"] = df["role_category"].apply(_normalize_role)

    df = df.dropna(subset=["job_id", "skill_name"])

    # Fill role_category from job_postings when missing
    role_from_job = df_postings.set_index("job_id")["role_category"].to_dict()
    df["role_category"] = df.apply(
        lambda r: r["role_category"] if r["role_category"] else role_from_job.get(r["job_id"]), axis=1
    )

    df = df.dropna(subset=["role_category"])

    # Ensure skill exists in skill master for that role
    valid_pairs = set(zip(df_skill_master["role"], df_skill_master["skill_name"]))
    df = df[df.apply(lambda r: (r["role_category"], r["skill_name"]) in valid_pairs, axis=1)]

    # Overwrite skill_category from master
    master_cat = df_skill_master.set_index(["role", "skill_name"])["skill_category"].to_dict()
    df["skill_category"] = df.apply(
        lambda r: master_cat.get((r["role_category"], r["skill_name"]), r["skill_category"]), axis=1
    )

    df = df.drop_duplicates(subset=["job_id", "skill_name"])

    return df.reset_index(drop=True)


def _clean_user_skills(df_user_skills, df_skill_master):
    df = df_user_skills.copy()

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["target_role"] = df["target_role"].apply(_normalize_role)
    df["skill_name"] = df["skill_name"].apply(_normalize_skill_name)

    def _clean_proficiency(value):
        if isinstance(value, str) and value.endswith("%"):
            value = value.replace("%", "")
        return pd.to_numeric(value, errors="coerce")

    df["proficiency_level"] = df["proficiency_level"].apply(_clean_proficiency).clip(0, 100)

    df = df.dropna(subset=["user_id", "skill_name", "proficiency_level"])

    # Fill target_role from skill master if missing (fallback to most common role)
    if df["target_role"].isna().any():
        mode_role = df["target_role"].mode(dropna=True)
        df["target_role"] = df["target_role"].fillna(mode_role.iloc[0] if not mode_role.empty else "Data Analyst")

    # Filter skills by master list
    valid_pairs = set(zip(df_skill_master["role"], df_skill_master["skill_name"]))
    df = df[df.apply(lambda r: (r["target_role"], r["skill_name"]) in valid_pairs, axis=1)]

    df = df.drop_duplicates(subset=["user_id", "skill_name"])

    return df.reset_index(drop=True)


def _build_role_skill_maps(df_skill_master):
    role_skills = {}
    role_importance = {}
    role_hours = {}

    for role in df_skill_master["role"].unique():
        subset = df_skill_master[df_skill_master["role"] == role]
        role_skills[role] = subset["skill_name"].tolist()
        role_importance[role] = dict(zip(subset["skill_name"], subset["importance_score"]))
        role_hours[role] = dict(zip(subset["skill_name"], subset["avg_learning_hours"]))

    return role_skills, role_importance, role_hours


def _compute_gap_score(proficiency, role, role_importance, scenario):
    multiplier_map = {
        "Normal": 1.0,
        "Kompetitif": 1.15,
        "Booming": 0.90,
    }
    multiplier = multiplier_map.get(scenario, 1.0)

    weighted_sum = 0
    importance_sum = 0
    for skill_name, importance in role_importance[role].items():
        user_level = proficiency.get(skill_name, 0) / 100.0
        importance = min(1.0, importance * multiplier)
        weighted_sum += user_level * importance
        importance_sum += importance

    if importance_sum == 0:
        return 1.0

    return round(1 - (weighted_sum / importance_sum), 4)


def _find_top_missing_skills(proficiency, role, role_importance, n=3):
    gaps = []
    for skill_name, importance in role_importance[role].items():
        user_level = proficiency.get(skill_name, 0) / 100.0
        gap = importance * (1 - user_level)
        gaps.append((skill_name, gap))

    gaps.sort(key=lambda x: x[1], reverse=True)
    return [g[0] for g in gaps[:n]]


def _estimate_weeks_ready(proficiency, role, role_importance, role_hours, study_hours, scenario):
    multiplier_map = {
        "Normal": 1.0,
        "Kompetitif": 1.3,
        "Booming": 0.8,
    }
    multiplier = multiplier_map.get(scenario, 1.0)

    total_hours = 0
    for skill_name, importance in role_importance[role].items():
        user_level = proficiency.get(skill_name, 0) / 100.0
        gap = max(0, 1 - user_level)
        total_hours += gap * role_hours[role].get(skill_name, 0) * importance

    if study_hours <= 0:
        return 999

    weeks = (total_hours / study_hours) * multiplier
    return round(weeks, 1)


def _readiness_label(gap_score):
    if gap_score <= 0.2:
        return "Ready"
    if gap_score <= 0.4:
        return "Almost Ready"
    if gap_score <= 0.6:
        return "Needs Work"
    if gap_score <= 0.8:
        return "Significant Gap"
    return "Major Gap"


def _clean_user_profiles(df_profiles, df_user_skills, df_skill_master):
    df = df_profiles.copy()

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["target_role"] = df["target_role"].apply(_normalize_role)
    df["background_level"] = df["background_level"].apply(_normalize_background)
    df["market_scenario"] = df["market_scenario"].apply(_normalize_scenario)

    df["study_hours_per_week"] = pd.to_numeric(df["study_hours_per_week"], errors="coerce")
    df["study_hours_per_week"] = df["study_hours_per_week"].clip(3, 25)

    df = df.dropna(subset=["user_id"])

    for col, default in [("target_role", "Data Analyst"), ("background_level", "Pemula"), ("market_scenario", "Normal")]:
        mode_value = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode_value.iloc[0] if not mode_value.empty else default)

    df["study_hours_per_week"] = df["study_hours_per_week"].fillna(df["study_hours_per_week"].median())

    # Build skill maps
    role_skills, role_importance, role_hours = _build_role_skill_maps(df_skill_master)

    # Recompute metrics using user_skills
    skill_map = {}
    for _, row in df_user_skills.iterrows():
        skill_map.setdefault(row["user_id"], {})[row["skill_name"]] = row["proficiency_level"]

    cleaned_records = []
    for _, row in df.iterrows():
        role = row["target_role"]
        if role not in role_skills:
            role = "Data Analyst"

        proficiency = {s: 0 for s in role_skills[role]}
        proficiency.update(skill_map.get(row["user_id"], {}))

        gap_score = _compute_gap_score(proficiency, role, role_importance, row["market_scenario"])
        top_missing = _find_top_missing_skills(proficiency, role, role_importance, n=3)
        est_weeks = _estimate_weeks_ready(
            proficiency, role, role_importance, role_hours, row["study_hours_per_week"], row["market_scenario"]
        )
        readiness = _readiness_label(gap_score)

        cleaned_records.append({
            "user_id": row["user_id"],
            "target_role": role,
            "background_level": row["background_level"],
            "study_hours_per_week": round(float(row["study_hours_per_week"]), 1),
            "market_scenario": row["market_scenario"],
            "gap_score": gap_score,
            "readiness_label": readiness,
            "top_missing_skill_1": top_missing[0],
            "top_missing_skill_2": top_missing[1],
            "top_missing_skill_3": top_missing[2],
            "estimated_weeks_ready": est_weeks,
            "current_skills_json": json.dumps(proficiency),
        })

    return pd.DataFrame(cleaned_records)


def _build_skill_frequency(df_job_skills, df_job_postings):
    counts = df_job_skills.groupby(["role_category", "skill_name"]).size().reset_index(name="frequency")
    totals = df_job_postings.groupby("role_category").size().reset_index(name="total_postings")
    merged = counts.merge(totals, on="role_category", how="left")
    merged["frequency_pct"] = (merged["frequency"] / merged["total_postings"] * 100).round(2)
    return merged


def _build_modeling_dataset(df_user_profiles, df_user_skills, df_skill_master):
    # Unique skills across roles
    all_skills = sorted(df_skill_master["skill_name"].unique())

    # Build skill map
    skill_map = {}
    for _, row in df_user_skills.iterrows():
        skill_map.setdefault(row["user_id"], {})[row["skill_name"]] = row["proficiency_level"]

    rows = []
    for _, row in df_user_profiles.iterrows():
        record = row.to_dict()
        user_skills = skill_map.get(row["user_id"], {})
        for skill in all_skills:
            record[skill] = user_skills.get(skill, 0)
        rows.append(record)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("RUN CLEANING PIPELINE")
    print("=" * 60)

    df_postings_raw = _load_raw("job_postings_raw.csv")
    df_skills_raw = _load_raw("job_skills_raw.csv")
    df_master_raw = _load_raw("skill_master_raw.csv")
    df_users_raw = _load_raw("user_profiles_raw.csv")
    df_user_skills_raw = _load_raw("user_skills_raw.csv")

    print("[1/5] Cleaning skill master...")
    df_master_clean = _build_skill_master(df_master_raw)

    print("[2/5] Cleaning job postings...")
    df_postings_clean = _clean_job_postings(df_postings_raw)

    print("[3/5] Cleaning job skills...")
    df_job_skills_clean = _clean_job_skills(df_skills_raw, df_postings_clean, df_master_clean)

    print("[4/5] Cleaning user skills and profiles...")
    df_user_skills_clean = _clean_user_skills(df_user_skills_raw, df_master_clean)
    df_users_clean = _clean_user_profiles(df_users_raw, df_user_skills_clean, df_master_clean)

    print("[5/5] Building derived datasets...")
    df_skill_freq = _build_skill_frequency(df_job_skills_clean, df_postings_clean)
    df_modeling = _build_modeling_dataset(df_users_clean, df_user_skills_clean, df_master_clean)

    _save_processed(df_postings_clean, "job_postings_cleaned.csv")
    _save_processed(df_job_skills_clean, "job_skills_cleaned.csv")
    _save_processed(df_master_clean, "skill_master_cleaned.csv")
    _save_processed(df_users_clean, "user_profiles_cleaned.csv")
    _save_processed(df_user_skills_clean, "user_skills_cleaned.csv")
    _save_processed(df_skill_freq, "skill_frequency.csv")
    _save_processed(df_modeling, "modeling_dataset.csv")

    print("\n[SUCCESS] Cleaning pipeline selesai. Output ada di data/processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
