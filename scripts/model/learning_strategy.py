"""
learning_strategy.py
====================
Generator strategi belajar step-by-step berbasis output model.

Mengambil hasil prediksi (gap score, missing skills, estimated weeks)
dan menghasilkan roadmap belajar yang personal dan actionable.
"""

import os
import sys
import json
import math

# ============================================================
# LEARNING RESOURCE SUGGESTIONS
# ============================================================
RESOURCE_MAP = {
    # Data Analyst skills (keys match skill_master_cleaned.csv skill_name exactly)
    "Python": ["Codecademy Python", "Google Python Class", "Automate the Boring Stuff"],
    "SQL": ["SQLBolt", "Mode Analytics SQL Tutorial", "W3Schools SQL"],
    "Statistics": ["Khan Academy Statistics", "StatQuest YouTube", "Coursera Statistics"],
    "Data Modeling": ["Kimball Group", "Data Modeling Made Simple", "Vertabelo Academy"],
    "ETL Process": ["Apache Airflow Docs", "dbt Tutorial", "Fivetran University"],
    "Hypothesis Testing": ["Khan Academy", "Seeing Theory (Brown)", "StatQuest"],
    "Regression Analysis": ["Coursera ML by Andrew Ng", "StatQuest", "Kaggle Learn"],
    "A B Testing": ["Optimizely Academy", "Evan Miller Blog", "Google Optimize Guide"],
    "Excel": ["ExcelJet", "Chandoo.org", "Microsoft Excel Training"],
    "Tableau": ["Tableau Public Gallery", "Tableau eLearning", "Andy Kriebel Blog"],
    "Power Bi": ["Microsoft Learn Power BI", "SQLBI", "Guy in a Cube YouTube"],
    "Pandas": ["Pandas Documentation", "Kaggle Pandas Tutorial", "Real Python Pandas"],
    "Numpy": ["NumPy Quickstart", "Kaggle NumPy", "Real Python NumPy"],
    "Google Sheets": ["Google Sheets Training", "Ben Collins Blog", "Spreadsheet Class"],
    "Jupyter Notebook": ["Jupyter Docs", "Real Python Jupyter", "DataCamp Tutorial"],
    "Analytical Thinking": ["Coursera Critical Thinking", "HBR Thinking Articles", "Practice with Kaggle"],
    "Communication": ["Coursera Business Communication", "Toastmasters", "TED Talks"],
    "Problem Solving": ["HackerRank", "LeetCode", "Project Euler"],
    "Attention To Detail": ["Practice data cleaning projects", "Kaggle Competitions", "Code review practice"],
    "Business Acumen": ["HBR Articles", "Coursera Business Strategy", "Industry reports"],

    # Frontend Developer skills
    "HTML": ["MDN Web Docs", "freeCodeCamp", "W3Schools HTML"],
    "CSS": ["CSS-Tricks", "MDN CSS", "Kevin Powell YouTube"],
    "Javascript": ["JavaScript.info", "Eloquent JavaScript", "freeCodeCamp JS"],
    "Typescript": ["TypeScript Handbook", "Total TypeScript", "Exercism TypeScript"],
    "React": ["React.dev Tutorial", "Scrimba React", "Kent C. Dodds Blog"],
    "Vue.Js": ["Vue.js Guide", "Vue Mastery", "Vue School"],
    "Rest Api": ["REST API Tutorial", "Postman Learning", "MDN Fetch API"],
    "Git": ["Pro Git Book", "Git Branching Game", "Atlassian Git Tutorial"],
    "Responsive Design": ["MDN Responsive Design", "CSS-Tricks Flexbox/Grid", "Kevin Powell"],
    "Web Performance": ["web.dev Performance", "Lighthouse", "Chrome DevTools Docs"],
    "Vs Code": ["VS Code Tips & Tricks", "VS Code Docs", "Fireship YouTube"],
    "Chrome Devtools": ["Chrome DevTools Docs", "Debugging JS Tutorial", "Network Analysis"],
    "Webpack": ["Webpack Docs", "Webpack Academy", "SurviveJS Webpack"],
    "Npm Yarn": ["npm Docs", "Yarn Docs", "Node.js Guides"],
    "Figma (Basic)": ["Figma Basics Tutorial", "Figma Community", "CharliMarieTV"],
    "Github": ["GitHub Skills", "GitHub Docs", "Git Immersion"],
    "Collaboration": ["Agile/Scrum Guide", "Atlassian Teamwork", "Team communication courses"],
    "Time Management": ["Pomodoro Technique", "Getting Things Done", "Todoist Productivity Methods"],

    # UI/UX Designer skills
    "User Research": ["NN/g UX Research", "Interaction Design Foundation", "UX Research Field Guide"],
    "Wireframing": ["Balsamiq Tutorials", "Figma Wireframing", "Sketch Wireframing"],
    "Prototyping": ["Figma Prototyping", "InVision Guide", "Principle App Tutorials"],
    "Usability Testing": ["NN/g Usability", "Maze Guides", "Steve Krug Don't Make Me Think"],
    "Information Architecture": ["IA Institute", "Abby Covert How to Make Sense", "Card Sorting Guide"],
    "Interaction Design": ["IxDF Courses", "Laws of UX", "Microinteractions Book"],
    "Design Systems": ["Design Systems Handbook", "Storybook Docs", "Material Design"],
    "Accessibility": ["WebAIM", "A11y Project", "WCAG Guidelines"],
    "Figma": ["Figma Tutorials", "Figma Community", "DesignCourse YouTube"],
    "Adobe Xd": ["Adobe XD Tutorials", "XD Resources", "Howard Pinsky YouTube"],
    "Sketch": ["Sketch Docs", "SketchTogether", "Sketch Resources"],
    "Miro": ["Miro Academy", "Miro Templates", "Miro YouTube"],
    "Invision": ["InVision Tutorials", "InVision Blog", "Design Better by InVision"],
    "Adobe Illustrator": ["Adobe Illustrator Tutorials", "Envato Tuts+", "Dansky YouTube"],
    "Maze": ["Maze Guides", "Maze Blog", "UX Testing with Maze"],
    "Empathy": ["Empathy Map Workshop", "Design Thinking", "IDEO Resources"],
    "Visual Thinking": ["Sketch Noting", "Visual Thinking by Dan Roam", "Graphic Recording"],
    "Creativity": ["Creative Confidence (IDEO)", "Lateral Thinking", "Design Sprints"],
}


def generate_learning_strategy(prediction_result):
    """
    Generate strategi belajar step-by-step berdasarkan output model.
    
    Parameters:
    -----------
    prediction_result : dict
        Output dari inference.predict_user()
    
    Returns:
    --------
    dict with:
        - summary: ringkasan strategi
        - phases: list of learning phases
        - weekly_plan: weekly breakdown
        - milestones: list of milestones
    """
    
    input_info = prediction_result['input_summary']
    gap_score = prediction_result['gap_score']
    readiness = prediction_result['readiness_label']
    est_weeks = prediction_result['estimated_weeks']
    all_gaps = prediction_result['all_skill_gaps']
    top_missing = prediction_result['top_missing_skills']
    study_hours = input_info['study_hours_per_week']
    
    # ---- Phase Planning ----
    # Bagi skill gaps menjadi 3 fase berdasarkan prioritas
    # Fase 1: Critical skills (gap tertinggi, importance tinggi)
    # Fase 2: Important skills (gap medium)
    # Fase 3: Nice-to-have skills (gap rendah)
    
    sorted_gaps = sorted(all_gaps, key=lambda x: x['gap'], reverse=True)
    
    # Split into phases
    n_skills = len(sorted_gaps)
    phase1_skills = sorted_gaps[:max(3, n_skills // 3)]
    phase2_skills = sorted_gaps[max(3, n_skills // 3):2 * n_skills // 3]
    phase3_skills = sorted_gaps[2 * n_skills // 3:]
    
    def calculate_phase_weeks(skills, study_hrs):
        """Hitung estimasi minggu untuk satu fase."""
        total_hours = sum(
            s['gap'] * s['avg_learning_hours'] 
            for s in skills if s['gap'] > 0.05
        )
        if study_hrs <= 0:
            return 0
        return max(1, round(total_hours / study_hrs))
    
    phase1_weeks = calculate_phase_weeks(phase1_skills, study_hours)
    phase2_weeks = calculate_phase_weeks(phase2_skills, study_hours)
    phase3_weeks = calculate_phase_weeks(phase3_skills, study_hours)
    
    # ---- Build Phases ----
    phases = []
    
    # Phase 1: Foundation & Critical Skills
    phase1_items = []
    for skill in phase1_skills:
        if skill['gap'] > 0.05:
            hours_needed = round(skill['gap'] * skill['avg_learning_hours'])
            resources = RESOURCE_MAP.get(skill['skill_name'], ["Cari kursus online"])
            phase1_items.append({
                'skill_name': skill['skill_name'],
                'category': skill['category'],
                'current_proficiency': skill['proficiency'],
                'target_proficiency': min(100, skill['proficiency'] + round(skill['gap'] * 100)),
                'hours_needed': hours_needed,
                'gap_severity': 'Critical' if skill['gap'] > 0.5 else 'High',
                'resources': resources[:2],
            })
    
    phases.append({
        'phase': 1,
        'name': 'Phase 1: Foundation & Critical Skills',
        'description': 'Fokus pada skill dengan gap terbesar dan importance tertinggi.',
        'estimated_weeks': phase1_weeks,
        'skills': phase1_items,
    })
    
    # Phase 2: Skill Deepening
    phase2_items = []
    for skill in phase2_skills:
        if skill['gap'] > 0.05:
            hours_needed = round(skill['gap'] * skill['avg_learning_hours'])
            resources = RESOURCE_MAP.get(skill['skill_name'], ["Cari kursus online"])
            phase2_items.append({
                'skill_name': skill['skill_name'],
                'category': skill['category'],
                'current_proficiency': skill['proficiency'],
                'target_proficiency': min(100, skill['proficiency'] + round(skill['gap'] * 100)),
                'hours_needed': hours_needed,
                'gap_severity': 'Medium',
                'resources': resources[:2],
            })
    
    phases.append({
        'phase': 2,
        'name': 'Phase 2: Skill Deepening',
        'description': 'Perkuat skill yang sudah ada dasar-dasarnya.',
        'estimated_weeks': phase2_weeks,
        'skills': phase2_items,
    })
    
    # Phase 3: Polish & Soft Skills
    phase3_items = []
    for skill in phase3_skills:
        if skill['gap'] > 0.05:
            hours_needed = round(skill['gap'] * skill['avg_learning_hours'])
            resources = RESOURCE_MAP.get(skill['skill_name'], ["Cari kursus online"])
            phase3_items.append({
                'skill_name': skill['skill_name'],
                'category': skill['category'],
                'current_proficiency': skill['proficiency'],
                'target_proficiency': min(100, skill['proficiency'] + round(skill['gap'] * 100)),
                'hours_needed': hours_needed,
                'gap_severity': 'Low',
                'resources': resources[:2],
            })
    
    phases.append({
        'phase': 3,
        'name': 'Phase 3: Polish & Soft Skills',
        'description': 'Lengkapi kompetensi dengan skill pelengkap.',
        'estimated_weeks': phase3_weeks,
        'skills': phase3_items,
    })
    
    # ---- Weekly Plan (first 4 weeks detail) ----
    weekly_plan = []
    all_active_skills = phase1_items + phase2_items + phase3_items
    
    if all_active_skills:
        for week in range(1, 5):
            n_skills = len(all_active_skills)
            idx1 = ((week - 1) * 2) % n_skills
            idx2 = ((week - 1) * 2 + 1) % n_skills
            
            # Select skills for the week
            week_skills = [all_active_skills[idx1]]
            if n_skills > 1:
                week_skills.append(all_active_skills[idx2])
                
            weekly_plan.append({
                'week': week,
                'focus_skills': [s['skill_name'] for s in week_skills],
                'hours': study_hours,
                'activities': [
                    f"Pelajari {week_skills[0]['skill_name']}: {week_skills[0]['resources'][0]}" if week_skills else "Tinjau materi",
                    f"Praktikkan {week_skills[0]['skill_name']}" if week_skills else "Proyek kecil",
                    f"Pelajari {week_skills[1]['skill_name']}: {week_skills[1]['resources'][0]}" if len(week_skills) > 1 else "Tinjau & refleksi",
                ]
            })
    
    # ---- Milestones ----
    milestones = [
        {
            'week': phase1_weeks,
            'milestone': 'Phase 1 Completed: Critical Skills',
            'expected_gap_score': round(gap_score * 0.6, 2),
        },
        {
            'week': phase1_weeks + phase2_weeks,
            'milestone': 'Phase 2 Completed: Skill Deepening',
            'expected_gap_score': round(gap_score * 0.3, 2),
        },
        {
            'week': phase1_weeks + phase2_weeks + phase3_weeks,
            'milestone': 'Phase 3 Completed: Job-Ready!',
            'expected_gap_score': round(gap_score * 0.1, 2),
        },
    ]
    
    # ---- Summary ----
    strategy = {
        'summary': {
            'current_status': readiness,
            'gap_score': gap_score,
            'total_estimated_weeks': phase1_weeks + phase2_weeks + phase3_weeks,
            'model_estimated_weeks': est_weeks,
            'study_hours_per_week': study_hours,
            'target_role': input_info['target_role'],
            'market_scenario': input_info['market_scenario'],
            'total_skills_to_improve': sum(1 for s in sorted_gaps if s['gap'] > 0.05),
            'critical_skills': len(phase1_items),
        },
        'phases': phases,
        'weekly_plan': weekly_plan,
        'milestones': milestones,
    }
    
    return strategy


def print_strategy(strategy):
    """Pretty-print learning strategy."""
    
    summary = strategy['summary']
    
    print("\n" + "=" * 60)
    print("📚 RENCANA BELAJAR PERSONAL")
    print("=" * 60)
    
    print(f"\n  Target Role    : {summary['target_role']}")
    print(f"  Status Saat Ini: {summary['current_status']}")
    print(f"  Gap Score      : {summary['gap_score']:.4f}")
    print(f"  Estimasi Waktu : {summary['total_estimated_weeks']} minggu")
    print(f"  Jam Belajar    : {summary['study_hours_per_week']} jam/minggu")
    print(f"  Skills to Fix  : {summary['total_skills_to_improve']} skills")
    
    for phase in strategy['phases']:
        print(f"\n{'─' * 60}")
        print(f"📌 Phase {phase['phase']}: {phase['name']} ({phase['estimated_weeks']} minggu)")
        print(f"   {phase['description']}")
        
        if phase['skills']:
            for s in phase['skills']:
                bar_fill = "█" * (s['current_proficiency'] // 5)
                bar_empty = "░" * ((s['target_proficiency'] - s['current_proficiency']) // 5)
                print(f"\n   • {s['skill_name']} [{s['gap_severity']}]")
                print(f"     {bar_fill}{bar_empty} {s['current_proficiency']}→{s['target_proficiency']} "
                      f"({s['hours_needed']} jam)")
                print(f"     Resources: {', '.join(s['resources'])}")
    
    print(f"\n{'─' * 60}")
    print(f"\n🏁 Milestones:")
    for m in strategy['milestones']:
        print(f"   Minggu {m['week']:3d}: {m['milestone']} "
              f"(target gap: {m['expected_gap_score']})")
    
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    # Demo standalone
    sys.path.insert(0, os.path.dirname(__file__))
    from inference import predict_user, load_model_and_metadata
    
    print("Loading model...")
    model, metadata, df_skill_master = load_model_and_metadata()
    
    # Sample prediction
    skill_profs = {
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
    
    result = predict_user(
        "Data Analyst", "Pemula", "Normal", 10, skill_profs,
        model, metadata, df_skill_master,
    )
    
    strategy = generate_learning_strategy(result)
    print_strategy(strategy)
