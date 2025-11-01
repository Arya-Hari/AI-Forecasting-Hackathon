import pandas as pd

skills_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_occupations_normalized.csv") 
benchmarks_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_benchmark_collated_scores.csv")

benchmarks_df = benchmarks_df[['Skill', 'Collated_Score']]

skills_df['Skill'] = skills_df['Skill'].astype(str).str.lower().str.strip().str.replace(' ', '_')
benchmarks_df['Skill'] = benchmarks_df['Skill'].astype(str).str.lower().str.strip().str.replace(' ', '_')

merged_df = pd.merge(skills_df, benchmarks_df, on='Skill', how='left')
merged_df = merged_df.dropna(subset=['Collated_Score'])

print(merged_df.head())
merged_df.to_csv("skills_jobs_with_ai_scores.csv", index=False)