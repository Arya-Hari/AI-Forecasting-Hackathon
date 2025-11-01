import pandas as pd

# Load files
skills_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_occupations_normalized.csv") 
benchmarks_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_benchmark_collated_scores.csv")

# Keep only Skill and Collated_Score columns
benchmarks_df = benchmarks_df[['Skill', 'Collated_Score']]

# === FIX: Standardize Skill Column Naming for Merge ===
# Convert Skill column in both DFs to lowercase and replace spaces with underscores
skills_df['Skill'] = skills_df['Skill'].astype(str).str.lower().str.strip().str.replace(' ', '_')
benchmarks_df['Skill'] = benchmarks_df['Skill'].astype(str).str.lower().str.strip().str.replace(' ', '_')
# ======================================================

# Merge by Skill (now standardized)
merged_df = pd.merge(skills_df, benchmarks_df, on='Skill', how='left')

# Drop skills that don’t have benchmark data yet
# NOTE: This step is now much more effective, only dropping skills 
# that are genuinely missing from the benchmark list.
merged_df = merged_df.dropna(subset=['Collated_Score'])

# Preview
print(merged_df.head())

# Save merged dataset
merged_df.to_csv("skills_jobs_with_ai_scores.csv", index=False)