"""
generate_job_postings.py
========================
Script untuk menghasilkan dataset job postings sintetis yang realistis
berdasarkan pola data dari LinkedIn Job Postings 2024 (Kaggle).

Sumber referensi: 
- Kaggle: "1.3M LinkedIn Jobs & Skills (2024)" by asaniczka
- URL: https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024

Output: data/raw/job_postings_raw.csv, data/raw/job_skills_raw.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

# Set seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# SKILL TAXONOMY — 20 skills per role
# Berdasarkan riset frekuensi skill di LinkedIn job postings
# ============================================================

SKILL_TAXONOMY = {
    "Data Analyst": {
        "Technical": [
            {"skill": "Python", "importance": 0.92, "avg_learning_hours": 200, "category": "Technical"},
            {"skill": "SQL", "importance": 0.95, "avg_learning_hours": 120, "category": "Technical"},
            {"skill": "Statistics", "importance": 0.85, "avg_learning_hours": 180, "category": "Technical"},
            {"skill": "Data Modeling", "importance": 0.70, "avg_learning_hours": 150, "category": "Technical"},
            {"skill": "ETL Process", "importance": 0.65, "avg_learning_hours": 100, "category": "Technical"},
            {"skill": "Hypothesis Testing", "importance": 0.60, "avg_learning_hours": 80, "category": "Technical"},
            {"skill": "Regression Analysis", "importance": 0.58, "avg_learning_hours": 90, "category": "Technical"},
            {"skill": "A/B Testing", "importance": 0.55, "avg_learning_hours": 60, "category": "Technical"},
        ],
        "Tool": [
            {"skill": "Excel", "importance": 0.88, "avg_learning_hours": 80, "category": "Tool"},
            {"skill": "Tableau", "importance": 0.78, "avg_learning_hours": 100, "category": "Tool"},
            {"skill": "Power BI", "importance": 0.75, "avg_learning_hours": 100, "category": "Tool"},
            {"skill": "Pandas", "importance": 0.80, "avg_learning_hours": 60, "category": "Tool"},
            {"skill": "NumPy", "importance": 0.65, "avg_learning_hours": 40, "category": "Tool"},
            {"skill": "Google Sheets", "importance": 0.50, "avg_learning_hours": 30, "category": "Tool"},
            {"skill": "Jupyter Notebook", "importance": 0.60, "avg_learning_hours": 20, "category": "Tool"},
        ],
        "Soft Skill": [
            {"skill": "Analytical Thinking", "importance": 0.90, "avg_learning_hours": 100, "category": "Soft Skill"},
            {"skill": "Communication", "importance": 0.82, "avg_learning_hours": 80, "category": "Soft Skill"},
            {"skill": "Problem Solving", "importance": 0.85, "avg_learning_hours": 90, "category": "Soft Skill"},
            {"skill": "Attention to Detail", "importance": 0.75, "avg_learning_hours": 50, "category": "Soft Skill"},
            {"skill": "Business Acumen", "importance": 0.68, "avg_learning_hours": 120, "category": "Soft Skill"},
        ],
    },
    "Frontend Developer": {
        "Technical": [
            {"skill": "HTML", "importance": 0.95, "avg_learning_hours": 60, "category": "Technical"},
            {"skill": "CSS", "importance": 0.95, "avg_learning_hours": 80, "category": "Technical"},
            {"skill": "JavaScript", "importance": 0.98, "avg_learning_hours": 250, "category": "Technical"},
            {"skill": "TypeScript", "importance": 0.78, "avg_learning_hours": 100, "category": "Technical"},
            {"skill": "React", "importance": 0.88, "avg_learning_hours": 150, "category": "Technical"},
            {"skill": "Vue.js", "importance": 0.55, "avg_learning_hours": 120, "category": "Technical"},
            {"skill": "REST API", "importance": 0.80, "avg_learning_hours": 60, "category": "Technical"},
            {"skill": "Git", "importance": 0.90, "avg_learning_hours": 40, "category": "Technical"},
            {"skill": "Responsive Design", "importance": 0.85, "avg_learning_hours": 50, "category": "Technical"},
            {"skill": "Web Performance", "importance": 0.62, "avg_learning_hours": 80, "category": "Technical"},
        ],
        "Tool": [
            {"skill": "VS Code", "importance": 0.82, "avg_learning_hours": 20, "category": "Tool"},
            {"skill": "Chrome DevTools", "importance": 0.75, "avg_learning_hours": 30, "category": "Tool"},
            {"skill": "Webpack", "importance": 0.58, "avg_learning_hours": 50, "category": "Tool"},
            {"skill": "npm/yarn", "importance": 0.80, "avg_learning_hours": 25, "category": "Tool"},
            {"skill": "Figma (Basic)", "importance": 0.50, "avg_learning_hours": 40, "category": "Tool"},
            {"skill": "GitHub", "importance": 0.85, "avg_learning_hours": 30, "category": "Tool"},
        ],
        "Soft Skill": [
            {"skill": "Problem Solving", "importance": 0.88, "avg_learning_hours": 90, "category": "Soft Skill"},
            {"skill": "Collaboration", "importance": 0.80, "avg_learning_hours": 60, "category": "Soft Skill"},
            {"skill": "Attention to Detail", "importance": 0.75, "avg_learning_hours": 50, "category": "Soft Skill"},
            {"skill": "Time Management", "importance": 0.70, "avg_learning_hours": 40, "category": "Soft Skill"},
        ],
    },
    "UI/UX Designer": {
        "Technical": [
            {"skill": "User Research", "importance": 0.92, "avg_learning_hours": 120, "category": "Technical"},
            {"skill": "Wireframing", "importance": 0.90, "avg_learning_hours": 60, "category": "Technical"},
            {"skill": "Prototyping", "importance": 0.88, "avg_learning_hours": 80, "category": "Technical"},
            {"skill": "Usability Testing", "importance": 0.82, "avg_learning_hours": 70, "category": "Technical"},
            {"skill": "Information Architecture", "importance": 0.75, "avg_learning_hours": 90, "category": "Technical"},
            {"skill": "Interaction Design", "importance": 0.80, "avg_learning_hours": 100, "category": "Technical"},
            {"skill": "Design Systems", "importance": 0.72, "avg_learning_hours": 80, "category": "Technical"},
            {"skill": "Accessibility", "importance": 0.65, "avg_learning_hours": 50, "category": "Technical"},
        ],
        "Tool": [
            {"skill": "Figma", "importance": 0.95, "avg_learning_hours": 100, "category": "Tool"},
            {"skill": "Adobe XD", "importance": 0.68, "avg_learning_hours": 80, "category": "Tool"},
            {"skill": "Sketch", "importance": 0.55, "avg_learning_hours": 70, "category": "Tool"},
            {"skill": "Miro", "importance": 0.50, "avg_learning_hours": 20, "category": "Tool"},
            {"skill": "InVision", "importance": 0.45, "avg_learning_hours": 30, "category": "Tool"},
            {"skill": "Adobe Illustrator", "importance": 0.60, "avg_learning_hours": 120, "category": "Tool"},
            {"skill": "Maze", "importance": 0.40, "avg_learning_hours": 25, "category": "Tool"},
        ],
        "Soft Skill": [
            {"skill": "Empathy", "importance": 0.90, "avg_learning_hours": 60, "category": "Soft Skill"},
            {"skill": "Communication", "importance": 0.85, "avg_learning_hours": 80, "category": "Soft Skill"},
            {"skill": "Visual Thinking", "importance": 0.82, "avg_learning_hours": 70, "category": "Soft Skill"},
            {"skill": "Collaboration", "importance": 0.78, "avg_learning_hours": 60, "category": "Soft Skill"},
            {"skill": "Creativity", "importance": 0.88, "avg_learning_hours": 80, "category": "Soft Skill"},
        ],
    },
}

# ============================================================
# COMPANY & LOCATION DATA — untuk realisme
# ============================================================

COMPANIES = {
    "Data Analyst": [
        "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Spotify",
        "Tokopedia", "Gojek", "Shopee", "Grab", "Traveloka", "Bukalapak",
        "Bank BCA", "Bank Mandiri", "Telkom Indonesia", "Astra International",
        "Deloitte", "McKinsey", "PwC", "Accenture", "IBM", "Samsung",
        "Unilever", "P&G", "Coca-Cola", "DBS Bank", "OCBC", "Standard Chartered",
        "Flipkart", "Lazada", "OVO", "Dana", "LinkAja",
    ],
    "Frontend Developer": [
        "Google", "Microsoft", "Meta", "Twitter/X", "GitHub", "Vercel",
        "Tokopedia", "Gojek", "Shopee", "Grab", "Traveloka", "Bukalapak",
        "Tiket.com", "Blibli", "Ruangguru", "Zenius", "Kitabisa",
        "Netflix", "Spotify", "Airbnb", "Stripe", "Figma Inc.",
        "Atlassian", "GitLab", "Notion", "Canva", "Wise",
        "Accenture", "ThoughtWorks", "Agoda", "Sea Group", "ByteDance",
    ],
    "UI/UX Designer": [
        "Google", "Apple", "Microsoft", "Meta", "Airbnb", "Figma Inc.",
        "Tokopedia", "Gojek", "Shopee", "Grab", "Traveloka", "Bukalapak",
        "Ruangguru", "Zenius", "Tiket.com", "Kitabisa", "Halodoc",
        "Spotify", "Netflix", "Pinterest", "Canva", "Notion",
        "IBM", "SAP", "Adobe", "InVision", "Maze",
        "Accenture", "McKinsey Digital", "IDEO", "Frog Design", "Fjord",
    ],
}

LOCATIONS = [
    "Jakarta, Indonesia", "Bandung, Indonesia", "Surabaya, Indonesia",
    "Yogyakarta, Indonesia", "Bali, Indonesia", "Medan, Indonesia",
    "Singapore", "Kuala Lumpur, Malaysia", "Bangkok, Thailand",
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "London, UK", "Berlin, Germany", "Amsterdam, Netherlands",
    "Sydney, Australia", "Tokyo, Japan", "Seoul, South Korea",
    "Remote", "Hybrid - Jakarta", "Hybrid - Singapore",
]

JOB_LEVELS = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
JOB_LEVEL_WEIGHTS = [0.25, 0.30, 0.30, 0.10, 0.05]

JOB_TYPES = ["Full-time", "Part-time", "Contract", "Internship", "Freelance"]
JOB_TYPE_WEIGHTS = [0.60, 0.05, 0.15, 0.15, 0.05]

# Variasi job titles
JOB_TITLE_VARIATIONS = {
    "Data Analyst": [
        "Data Analyst", "Junior Data Analyst", "Senior Data Analyst",
        "Business Data Analyst", "Marketing Data Analyst", "Financial Data Analyst",
        "Data Analyst Intern", "Associate Data Analyst", "Lead Data Analyst",
        "BI Analyst", "Analytics Analyst", "Data Insights Analyst",
    ],
    "Frontend Developer": [
        "Frontend Developer", "Junior Frontend Developer", "Senior Frontend Developer",
        "Frontend Engineer", "React Developer", "Vue.js Developer",
        "Frontend Developer Intern", "Lead Frontend Developer", "UI Developer",
        "Web Developer", "JavaScript Developer", "Full Stack Developer (Frontend Focus)",
    ],
    "UI/UX Designer": [
        "UI/UX Designer", "Junior UI/UX Designer", "Senior UI/UX Designer",
        "UX Designer", "UI Designer", "Product Designer",
        "UX Researcher", "Interaction Designer", "Visual Designer",
        "Design Intern", "Lead Product Designer", "UX/UI Designer",
    ],
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

SKILL_TYPO_MAP = {
    "JavaScript": ["Java Script", "Javasript", "JS", "ReactJS"],
    "Python": ["Pyhton", "python ", "PYTHON"],
    "SQL": ["Sql", "S Q L"],
    "Power BI": ["PowerBI", "Power-BI", "poweR bi"],
    "Excel": ["Ms Excel", "MS-Excel", "Excell"],
    "A/B Testing": ["A/BTesting", "AB Testing"],
    "Machine Learning": ["Machine-Learning", "ML", "M L"],
    "User Research": ["UserResearch", "User-Research"],
    "Design Systems": ["DesignSystem", "Design-Systems"],
}

JOB_LEVEL_VARIANTS = [
    "Entry level", "entry-level", "Associate ", "mid level", "Mid-Senior",
    "Seniorr", "director", "executive ", "Intern", "n/a",
]

JOB_TYPE_VARIANTS = [
    "Full time", "Part time", "Contractor", "Intern", "Freelance ",
    "full-time", "contract", "n/a",
]

LOCATION_VARIANTS = [
    "jakarta, indonesia", "Jakarta, ID", " Bandung , Indonesia ",
    "remote", "HYBRID - JAKARTA", "Hybrid- Singapore",
]


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
    if random.random() < 0.2:
        text = text.replace("-", " ")
    if random.random() < 0.2:
        text = text.replace(",", " ,")
    return text


def _apply_dirty_job_postings(df_postings):
    df = df_postings.copy()
    n = len(df)

    # Missing values (moderate)
    for col in ["job_location", "job_level", "job_type", "first_seen"]:
        mask = _random_mask(n, DIRT_CONFIG["missing_rate"])
        df.loc[mask, col] = None

    # Job level/type variants
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "job_level"] = np.random.choice(JOB_LEVEL_VARIANTS, size=mask.sum())
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "job_type"] = np.random.choice(JOB_TYPE_VARIANTS, size=mask.sum())

    # Location variants
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "job_location"] = np.random.choice(LOCATION_VARIANTS, size=mask.sum())

    # Role category dirty strings
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "role_category"] = df.loc[mask, "role_category"].apply(_dirty_text)

    # Date format noise
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "first_seen"] = df.loc[mask, "first_seen"].apply(
        lambda x: x.replace("-", "/") if isinstance(x, str) else x
    )

    # Job ID missing/duplicate
    mask = _random_mask(n, DIRT_CONFIG["missing_rate"] / 2)
    df.loc[mask, "job_id"] = None
    dup_count = max(1, int(n * DIRT_CONFIG["duplicate_rate"]))
    dup_rows = df.sample(dup_count, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Near-duplicates (small text noise)
    near_count = max(1, int(n * DIRT_CONFIG["near_duplicate_rate"]))
    near_rows = df.sample(near_count, random_state=7).copy()
    near_rows["job_title"] = near_rows["job_title"].apply(_dirty_text)
    near_rows["company"] = near_rows["company"].apply(_dirty_text)
    df = pd.concat([df, near_rows], ignore_index=True)

    return df


def _apply_dirty_job_skills(df_skills):
    df = df_skills.copy()
    n = len(df)

    # Skill typo variants
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    for idx in df[mask].index:
        skill = df.at[idx, "skill_name"]
        if skill in SKILL_TYPO_MAP:
            df.at[idx, "skill_name"] = random.choice(SKILL_TYPO_MAP[skill])
        else:
            df.at[idx, "skill_name"] = _dirty_text(skill)

    # Skill category dirty strings
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "skill_category"] = df.loc[mask, "skill_category"].apply(_dirty_text)

    # Missing values
    for col in ["skill_name", "skill_category"]:
        mask = _random_mask(n, DIRT_CONFIG["missing_rate"] / 2)
        df.loc[mask, col] = None

    # Duplicates
    dup_count = max(1, int(n * DIRT_CONFIG["duplicate_rate"]))
    df = pd.concat([df, df.sample(dup_count, random_state=99)], ignore_index=True)

    return df


def _apply_dirty_skill_master(df_master):
    df = df_master.copy()
    n = len(df)

    # Importance outliers
    mask = _random_mask(n, DIRT_CONFIG["outlier_rate"])
    df.loc[mask, "importance_score"] = df.loc[mask, "importance_score"] * 1.5

    # Missing values
    for col in ["importance_score", "avg_learning_hours"]:
        mask = _random_mask(n, DIRT_CONFIG["missing_rate"] / 2)
        df.loc[mask, col] = None

    # Skill name noise
    mask = _random_mask(n, DIRT_CONFIG["typo_rate"])
    df.loc[mask, "skill_name"] = df.loc[mask, "skill_name"].apply(_dirty_text)

    # Duplicates
    dup_count = max(1, int(n * DIRT_CONFIG["duplicate_rate"]))
    df = pd.concat([df, df.sample(dup_count, random_state=123)], ignore_index=True)

    return df


def generate_job_postings(n_per_role=200):
    """Generate n job postings per role."""
    
    all_postings = []
    all_skills = []
    job_id = 1
    
    for role in ["Data Analyst", "Frontend Developer", "UI/UX Designer"]:
        print(f"  Generating {n_per_role} job postings untuk {role}...")
        
        # Kumpulkan semua skill untuk role ini
        role_skills = []
        for cat_skills in SKILL_TAXONOMY[role].values():
            role_skills.extend(cat_skills)
        
        for i in range(n_per_role):
            # Generate job metadata
            job_title = random.choice(JOB_TITLE_VARIATIONS[role])
            company = random.choice(COMPANIES[role])
            location = random.choice(LOCATIONS)
            job_level = np.random.choice(JOB_LEVELS, p=JOB_LEVEL_WEIGHTS)
            job_type = np.random.choice(JOB_TYPES, p=JOB_TYPE_WEIGHTS)
            
            # Generate date (2024)
            start_date = datetime(2024, 1, 1)
            random_days = random.randint(0, 364)
            first_seen = start_date + timedelta(days=random_days)
            
            posting = {
                "job_id": f"JOB-{job_id:05d}",
                "job_title": job_title,
                "company": company,
                "job_location": location,
                "role_category": role,
                "job_level": job_level,
                "job_type": job_type,
                "first_seen": first_seen.strftime("%Y-%m-%d"),
                "source": "Kaggle-LinkedIn-Synthetic",
            }
            all_postings.append(posting)
            
            # Generate skills untuk job ini (8-15 skills per posting)
            n_skills = random.randint(8, 15)
            # Weighted random selection berdasarkan importance
            skill_weights = np.array([s["importance"] for s in role_skills])
            skill_weights = skill_weights / skill_weights.sum()
            
            selected_indices = np.random.choice(
                len(role_skills), 
                size=min(n_skills, len(role_skills)), 
                replace=False,
                p=skill_weights
            )
            
            for idx in selected_indices:
                skill_entry = {
                    "job_id": f"JOB-{job_id:05d}",
                    "skill_name": role_skills[idx]["skill"],
                    "skill_category": role_skills[idx]["category"],
                    "role_category": role,
                }
                all_skills.append(skill_entry)
            
            job_id += 1
    
    df_postings = pd.DataFrame(all_postings)
    df_skills = pd.DataFrame(all_skills)
    
    return df_postings, df_skills


def generate_skill_master():
    """Generate skill master table dengan importance scores dan market demand."""
    
    records = []
    for role, categories in SKILL_TAXONOMY.items():
        for cat_name, skills in categories.items():
            for skill in skills:
                # Market demand variations
                base_importance = skill["importance"]
                record = {
                    "role": role,
                    "skill_name": skill["skill"],
                    "skill_category": skill["category"],
                    "importance_score": round(base_importance, 2),
                    "avg_learning_hours": skill["avg_learning_hours"],
                    # Market scenario adjustments
                    "market_demand_normal": round(base_importance, 2),
                    "market_demand_competitive": round(min(1.0, base_importance * 1.15), 2),
                    "market_demand_booming": round(min(1.0, base_importance * 0.90), 2),
                }
                records.append(record)
    
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("GENERATE JOB POSTINGS & SKILL MASTER DATA")
    print("=" * 60)
    
    # Paths
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Generate job postings (200 per role = 600 total)
    print("\n[1/3] Generating job postings...")
    df_postings, df_skills = generate_job_postings(n_per_role=200)
    df_postings = _apply_dirty_job_postings(df_postings)
    df_skills = _apply_dirty_job_skills(df_skills)
    
    postings_path = os.path.join(raw_dir, "job_postings_raw.csv")
    skills_raw_path = os.path.join(raw_dir, "job_skills_raw.csv")
    
    df_postings.to_csv(postings_path, index=False)
    df_skills.to_csv(skills_raw_path, index=False)
    
    print(f"  ✅ Job postings: {len(df_postings)} rows → {postings_path}")
    print(f"  ✅ Job skills: {len(df_skills)} rows → {skills_raw_path}")
    
    # 2. Generate skill master
    print("\n[2/3] Generating skill master table...")
    df_master = generate_skill_master()
    df_master = _apply_dirty_skill_master(df_master)
    master_path = os.path.join(raw_dir, "skill_master_raw.csv")
    df_master.to_csv(master_path, index=False)
    print(f"  ✅ Skill master: {len(df_master)} rows → {master_path}")
    
    # 3. Summary
    print("\n[3/3] Summary:")
    print(f"  • Total job postings: {len(df_postings)}")
    print(f"  • Total skill entries: {len(df_skills)}")
    print(f"  • Roles: {df_postings['role_category'].nunique()}")
    print(f"  • Unique companies: {df_postings['company'].nunique()}")
    print(f"  • Unique skills: {df_skills['skill_name'].nunique()}")
    print(f"  • Skill master entries: {len(df_master)}")
    
    for role in ["Data Analyst", "Frontend Developer", "UI/UX Designer"]:
        n = len(df_postings[df_postings["role_category"] == role])
        ns = len(df_skills[df_skills["role_category"] == role])
        print(f"  • {role}: {n} postings, {ns} skill entries")
    
    print("\n✅ Semua data raw berhasil di-generate!")
    print("=" * 60)


if __name__ == "__main__":
    main()
