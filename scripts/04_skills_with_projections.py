import pandas as pd
import numpy as np
from pathlib import Path
import warnings

CONFIG = {
    'job_skill_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_jobs_with_ai_scores.csv",
    'benchmarks_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_benchmark_collated_scores.csv",
    'output_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_with_projections.csv",
    'start_year': 2024,
    'end_year': 2035,
    'n_samples': 2000,
    'growth_sd_fraction': 0.20,
    'random_seed': 42,
    'fallback_growth': 9.0
}

BENCHMARK_GROWTH_RATES = {
    "MMLU": 9.5,
    "MMLU-Pro": 9.5,
    "MMLU-Pro(1yr-spike)": 33.1,
    "HumanEval": 16.2,
    "MATH": 18.2,
    "MedQA": 9.3,
    "HellaSwag": 9.6,
    "BIG-Bench": 10.0,
    "LegalBench": 9.0,
    "PubMedQA": 9.0
}

def standardize_name(series_or_string):
    if isinstance(series_or_string, pd.Series):
        return (series_or_string
                .astype(str)
                .str.lower()
                .str.strip()
                .str.replace(' ', '_')
                .str.replace('__+', '_', regex=True))
    else:
        if pd.isna(series_or_string):
            return series_or_string
        return str(series_or_string).lower().strip().replace(' ', '_')

def get_benchmark_growth(benchmark_name):
    if pd.isna(benchmark_name):
        return CONFIG['fallback_growth']
    standardized = standardize_name(benchmark_name)
    return BENCHMARK_GROWTH_RATES.get(standardized, CONFIG['fallback_growth'])

def compute_weighted_growth(row):
    total_similarity = 0.0
    weighted_sum = 0.0
    
    for i in range(1, 6):
        benchmark_col = f"Benchmark_{i}"
        similarity_col = f"Benchmark_{i}_Similarity"
        
        if benchmark_col in row.index and similarity_col in row.index:
            similarity = row[similarity_col]
            benchmark = row[benchmark_col]
            
            if not pd.isna(similarity) and similarity > 0:
                growth_rate = get_benchmark_growth(benchmark)
                weighted_sum += similarity * growth_rate
                total_similarity += similarity
    
    if total_similarity <= 0:
        return CONFIG['fallback_growth']
    
    return weighted_sum / total_similarity

def main():
    np.random.seed(CONFIG['random_seed'])
    
    jobskills = pd.read_csv(CONFIG['job_skill_path'])
    benchmarks = pd.read_csv(CONFIG['benchmarks_path'])
    benchmarks = benchmarks.copy()
    
    for i in range(1, 6):
        benchmark_col = f"Benchmark_{i}"
        if benchmark_col in benchmarks.columns:
            benchmarks[benchmark_col] = benchmarks[benchmark_col].apply(standardize_name)
    
    benchmarks['Skill'] = standardize_name(benchmarks['Skill'])
    similarity_cols = [c for c in benchmarks.columns if 'similarity' in c.lower()]
    benchmarks[similarity_cols] = benchmarks[similarity_cols].fillna(0.0)
    benchmarks['Annual_Growth_pts'] = benchmarks.apply(compute_weighted_growth, axis=1)
    skill_growth = benchmarks[['Skill', 'Annual_Growth_pts']].drop_duplicates()
    skill_growth = skill_growth.drop_duplicates(subset=['Skill'], keep='first')
    skill_growth = skill_growth.set_index('Skill')
    
    jobskills = jobskills.copy()
    jobskills['Skill_Standardized'] = standardize_name(jobskills['Skill'])
    jobskills = jobskills.merge(skill_growth, left_on='Skill_Standardized', right_index=True, how='left')
    jobskills['Annual_Growth_pts'] = jobskills['Annual_Growth_pts'].fillna(CONFIG['fallback_growth'])
    jobskills = jobskills.drop(columns=['Skill_Standardized'])
    jobskills['AI_Score_2024_raw'] = pd.to_numeric(jobskills['Collated_Score'], errors='coerce')
    jobskills['AI_Score_2024'] = (jobskills['AI_Score_2024_raw'] / 100.0).clip(0.0, 1.0)
    
    years = list(range(CONFIG['start_year'], CONFIG['end_year'] + 1))
    jobskills['Annual_Growth_frac'] = jobskills['Annual_Growth_pts'] / 100.0
    
    for year in years:
        delta_years = year - CONFIG['start_year']
        jobskills[f'AI_{year}'] = (
            jobskills['AI_Score_2024'] + delta_years * jobskills['Annual_Growth_frac']
        ).clip(0.0, 1.0)
    
    unique_skills = jobskills['Skill'].unique()
    skill_data = jobskills.groupby('Skill').first()
    base_scores = skill_data['AI_Score_2024'].values
    growth_rates_frac = skill_data['Annual_Growth_frac'].values
    growth_sds_frac = np.maximum(0.0001, np.abs(growth_rates_frac) * CONFIG['growth_sd_fraction'])
    sampled_growth = np.random.normal(loc=growth_rates_frac, scale=growth_sds_frac, size=(CONFIG['n_samples'], len(unique_skills)))
    sampled_growth = np.clip(sampled_growth, -0.05, 1.0)
    
    skill_to_idx = {skill: i for i, skill in enumerate(unique_skills)}
    
    for year in years:
        delta_years = year - CONFIG['start_year']
        projected_scores = np.clip(base_scores + sampled_growth * delta_years, 0.0, 1.0)
        row_skill_indices = jobskills['Skill'].map(skill_to_idx).values
        row_distributions = projected_scores[:, row_skill_indices]
        jobskills[f'AI_{year}_median'] = np.percentile(row_distributions, 50, axis=0)
        jobskills[f'AI_{year}_p05'] = np.percentile(row_distributions, 5, axis=0)
        jobskills[f'AI_{year}_p95'] = np.percentile(row_distributions, 95, axis=0)
    
    base_cols = ['Occupation', 'Code', 'Skill', 'Importance', 'Level', 'SkillDemand', 'Collated_Score', 'AI_Score_2024', 'Annual_Growth_pts']
    median_cols = [f'AI_{y}_median' for y in years]
    p05_cols = [f'AI_{y}_p05' for y in years]
    p95_cols = [f'AI_{y}_p95' for y in years]
    output_cols = [c for c in base_cols if c in jobskills.columns] + median_cols + p05_cols + p95_cols
    jobskills[output_cols].to_csv(CONFIG['output_path'], index=False)

if __name__ == "__main__":
    main()