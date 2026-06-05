import os
import numpy as np
import pandas as pd
from inference import predict_user, load_model_and_metadata

def main():
    model, metadata, df_skill_master = load_model_and_metadata()
    
    # Get role skills for Data Analyst
    role_skills = set(df_skill_master[df_skill_master['role'] == "Data Analyst"]['skill_name'].unique())
    
    # 1. Pemula Profile
    skills_pemula = {s: (25 if s in role_skills else 0) for s in metadata['skill_columns']}
    res_pemula = predict_user(
        target_role="Data Analyst",
        background_level="Pemula",
        market_scenario="Normal",
        study_hours_per_week=10,
        skill_proficiencies=skills_pemula,
        model=model, metadata=metadata, df_skill_master=df_skill_master
    )
    
    # 2. Menengah Profile
    skills_menengah = {s: (55 if s in role_skills else 0) for s in metadata['skill_columns']}
    res_menengah = predict_user(
        target_role="Data Analyst",
        background_level="Menengah",
        market_scenario="Normal",
        study_hours_per_week=10,
        skill_proficiencies=skills_menengah,
        model=model, metadata=metadata, df_skill_master=df_skill_master
    )
    
    # 3. Lanjutan Profile
    skills_lanjut = {s: (78 if s in role_skills else 0) for s in metadata['skill_columns']}
    res_lanjut = predict_user(
        target_role="Data Analyst",
        background_level="Lanjutan",
        market_scenario="Normal",
        study_hours_per_week=10,
        skill_proficiencies=skills_lanjut,
        model=model, metadata=metadata, df_skill_master=df_skill_master
    )
    
    print("\n" + "=" * 60)
    print("MODEL PREDICTIONS TEST")
    print("=" * 60)
    print(f"Pemula (Skills=25):   Predicted Gap = {res_pemula['gap_score']:.4f} ({res_pemula['readiness_label']}), Weeks = {res_pemula['estimated_weeks']:.1f}")
    print(f"Menengah (Skills=55): Predicted Gap = {res_menengah['gap_score']:.4f} ({res_menengah['readiness_label']}), Weeks = {res_menengah['estimated_weeks']:.1f}")
    print(f"Lanjutan (Skills=78): Predicted Gap = {res_lanjut['gap_score']:.4f} ({res_lanjut['readiness_label']}), Weeks = {res_lanjut['estimated_weeks']:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
