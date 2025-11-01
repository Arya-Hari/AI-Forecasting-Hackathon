import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# --- CONFIGURATION ---
CONFIG = {
    'job_skill_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_jobs_with_ai_scores.csv",
    'benchmarks_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_benchmark_collated_scores.csv",
    'output_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_with_projections.csv",
    'start_year': 2024,
    'end_year': 2035,
    'n_samples': 2000,
    'growth_sd_fraction': 0.20,  # 20% of growth rate as std deviation
    'random_seed': 42,
    'fallback_growth': 9.0  # pts/year
}

# Benchmark annual growth rates (percentage points per year)
BENCHMARK_GROWTH_RATES = {
    # Core benchmarks
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

# --- HELPER FUNCTIONS ---

def standardize_name(series_or_string):
    """Standardize skill/benchmark names for consistent matching."""
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
    """Get growth rate for a benchmark, with fallback."""
    if pd.isna(benchmark_name):
        return CONFIG['fallback_growth']
    standardized = standardize_name(benchmark_name)
    return BENCHMARK_GROWTH_RATES.get(standardized, CONFIG['fallback_growth'])

def compute_weighted_growth(row):
    """
    Compute weighted average growth rate based on benchmark similarities.
    Uses all Benchmark_N and Benchmark_N_Similarity columns.
    """
    total_similarity = 0.0
    weighted_sum = 0.0
    
    for i in range(1, 6):  # Benchmark_1 through Benchmark_5
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

# --- MAIN PROCESSING ---

def main():
    print("=" * 60)
    print("AI SKILL CAPABILITY PROJECTION MODEL")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(CONFIG['random_seed'])
    print(f"\n🎲 Random seed set to {CONFIG['random_seed']} for reproducibility")
    
    # --- 1. LOAD DATA ---
    print("\n📂 Loading data...")
    try:
        jobskills = pd.read_csv(CONFIG['job_skill_path'])
        print(f"   ✅ Job-skills data: {len(jobskills):,} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"Job-skills file not found: {CONFIG['job_skill_path']}")
    
    try:
        benchmarks = pd.read_csv(CONFIG['benchmarks_path'])
        print(f"   ✅ Benchmarks data: {len(benchmarks):,} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"Benchmarks file not found: {CONFIG['benchmarks_path']}")
    
    # --- 2. PREPARE BENCHMARKS DATA ---
    print("\n🔧 Processing benchmark data...")
    benchmarks = benchmarks.copy()
    
    # Standardize benchmark names in all columns
    for i in range(1, 6):
        benchmark_col = f"Benchmark_{i}"
        if benchmark_col in benchmarks.columns:
            benchmarks[benchmark_col] = benchmarks[benchmark_col].apply(standardize_name)
    
    # Standardize skill names
    benchmarks['Skill'] = standardize_name(benchmarks['Skill'])
    
    # Fill missing similarities with 0
    similarity_cols = [c for c in benchmarks.columns if 'similarity' in c.lower()]
    benchmarks[similarity_cols] = benchmarks[similarity_cols].fillna(0.0)
    print(f"   Found {len(similarity_cols)} similarity columns")
    
    # Compute weighted growth rate for each skill
    print("   • Computing weighted growth rates...")
    benchmarks['Annual_Growth_pts'] = benchmarks.apply(compute_weighted_growth, axis=1)
    
    # Summary statistics
    print(f"\n   Growth rate statistics (pts/year):")
    print(f"      Mean: {benchmarks['Annual_Growth_pts'].mean():.2f}")
    print(f"      Median: {benchmarks['Annual_Growth_pts'].median():.2f}")
    print(f"      Range: [{benchmarks['Annual_Growth_pts'].min():.2f}, {benchmarks['Annual_Growth_pts'].max():.2f}]")
    
    # Create skill-growth lookup
    skill_growth = benchmarks[['Skill', 'Annual_Growth_pts']].drop_duplicates()
    
    # Check for duplicates after standardization
    dupes = skill_growth['Skill'].duplicated().sum()
    if dupes > 0:
        print(f"   ⚠️  Warning: {dupes} duplicate skills after standardization, keeping first")
        skill_growth = skill_growth.drop_duplicates(subset=['Skill'], keep='first')
    
    skill_growth = skill_growth.set_index('Skill')
    
    # --- 3. MERGE GROWTH RATES WITH JOB-SKILLS ---
    print("\n🔗 Merging growth rates with job-skills data...")
    jobskills = jobskills.copy()
    
    # Standardize skill names in jobskills
    jobskills['Skill_Standardized'] = standardize_name(jobskills['Skill'])
    
    # Pre-merge analysis
    skills_in_jobs = set(jobskills['Skill_Standardized'].unique())
    skills_in_benchmarks = set(skill_growth.index)
    matched = skills_in_jobs & skills_in_benchmarks
    unmatched = skills_in_jobs - skills_in_benchmarks
    
    print(f"   Skills in job data: {len(skills_in_jobs)}")
    print(f"   Skills in benchmarks: {len(skills_in_benchmarks)}")
    print(f"   Matched skills: {len(matched)} ({len(matched)/len(skills_in_jobs)*100:.1f}%)")
    print(f"   Unmatched skills: {len(unmatched)} ({len(unmatched)/len(skills_in_jobs)*100:.1f}%)")
    
    if unmatched and len(unmatched) <= 10:
        print(f"   ⚠️  Unmatched: {sorted(unmatched)}")
    
    # Merge
    jobskills = jobskills.merge(skill_growth, left_on='Skill_Standardized', right_index=True, how='left')
    
    # Fill missing with fallback
    missing_count = jobskills['Annual_Growth_pts'].isna().sum()
    if missing_count > 0:
        print(f"   ⚠️  Filling {missing_count:,} rows ({missing_count/len(jobskills)*100:.1f}%) with fallback growth: {CONFIG['fallback_growth']} pts/year")
        jobskills['Annual_Growth_pts'] = jobskills['Annual_Growth_pts'].fillna(CONFIG['fallback_growth'])
    
    jobskills = jobskills.drop(columns=['Skill_Standardized'])
    
    # --- 4. NORMALIZE CURRENT AI SCORES ---
    print("\n📊 Normalizing AI scores...")
    jobskills['AI_Score_2024_raw'] = pd.to_numeric(jobskills['Collated_Score'], errors='coerce')
    jobskills['AI_Score_2024'] = (jobskills['AI_Score_2024_raw'] / 100.0).clip(0.0, 1.0)
    
    print(f"   Score range: [{jobskills['AI_Score_2024'].min():.3f}, {jobskills['AI_Score_2024'].max():.3f}]")
    print(f"   Mean score: {jobskills['AI_Score_2024'].mean():.3f}")
    
    # --- 5. GENERATE PROJECTIONS ---
    years = list(range(CONFIG['start_year'], CONFIG['end_year'] + 1))
    print(f"\n📈 Generating projections for {len(years)} years ({CONFIG['start_year']}-{CONFIG['end_year']})...")
    
    # Convert growth from percentage points to fraction
    jobskills['Annual_Growth_frac'] = jobskills['Annual_Growth_pts'] / 100.0
    
    # Deterministic projections
    print("   • Creating deterministic projections...")
    for year in years:
        delta_years = year - CONFIG['start_year']
        jobskills[f'AI_{year}'] = (
            jobskills['AI_Score_2024'] + delta_years * jobskills['Annual_Growth_frac']
        ).clip(0.0, 1.0)
    
    # --- 6. MONTE CARLO SIMULATION ---
    print(f"\n🎲 Running Monte Carlo simulation ({CONFIG['n_samples']:,} samples)...")
    
    # Process at skill level for efficiency
    unique_skills = jobskills['Skill'].unique()
    n_skills = len(unique_skills)
    
    print(f"   Processing {n_skills} unique skills...")
    
    # Extract skill-level parameters
    skill_data = jobskills.groupby('Skill').first()
    base_scores = skill_data['AI_Score_2024'].values
    growth_rates_frac = skill_data['Annual_Growth_frac'].values
    
    # Standard deviation: fraction of growth rate (minimum 0.01% to avoid zero variance)
    growth_sds_frac = np.maximum(
        0.0001,  # minimum 0.01% std dev
        np.abs(growth_rates_frac) * CONFIG['growth_sd_fraction']
    )
    
    # Sample growth rates for all skills (N_SAMPLES x n_skills)
    print("   • Sampling growth rates...")
    sampled_growth = np.random.normal(
        loc=growth_rates_frac,
        scale=growth_sds_frac,
        size=(CONFIG['n_samples'], n_skills)
    )
    
    # Clip to reasonable bounds (-5% to +100% per year)
    sampled_growth = np.clip(sampled_growth, -0.05, 1.0)
    
    # Compute projections for each year
    print("   • Computing confidence intervals...")
    skill_to_idx = {skill: i for i, skill in enumerate(unique_skills)}
    
    for year in years:
        delta_years = year - CONFIG['start_year']
        
        # Projected scores: (N_SAMPLES x n_skills)
        projected_scores = np.clip(
            base_scores + sampled_growth * delta_years,
            0.0,
            1.0
        )
        
        # Map skill-level results to rows
        row_skill_indices = jobskills['Skill'].map(skill_to_idx).values
        row_distributions = projected_scores[:, row_skill_indices]  # (N_SAMPLES x n_rows)
        
        # Compute percentiles across samples (axis=0)
        jobskills[f'AI_{year}_median'] = np.percentile(row_distributions, 50, axis=0)
        jobskills[f'AI_{year}_p05'] = np.percentile(row_distributions, 5, axis=0)
        jobskills[f'AI_{year}_p95'] = np.percentile(row_distributions, 95, axis=0)
    
    # --- 7. SAVE RESULTS ---
    print(f"\n💾 Saving results to {CONFIG['output_path']}...")
    
    # Define output columns
    base_cols = ['Occupation', 'Code', 'Skill', 'Importance', 'Level', 
                 'SkillDemand', 'Collated_Score', 'AI_Score_2024', 'Annual_Growth_pts']
    
    # Add projection columns
    median_cols = [f'AI_{y}_median' for y in years]
    p05_cols = [f'AI_{y}_p05' for y in years]
    p95_cols = [f'AI_{y}_p95' for y in years]
    
    # Filter to only existing columns
    output_cols = [c for c in base_cols if c in jobskills.columns] + median_cols + p05_cols + p95_cols
    
    jobskills[output_cols].to_csv(CONFIG['output_path'], index=False)
    print(f"   ✅ Saved {len(jobskills):,} rows with {len(output_cols)} columns")
    
    # --- 8. SUMMARY STATISTICS ---
    print("\n📊 Projection Summary:")
    print(f"\n   2024 (baseline):")
    print(f"      Mean: {jobskills['AI_Score_2024'].mean():.3f}")
    
    end_year = CONFIG['end_year']
    print(f"\n   {end_year} (projected):")
    print(f"      Mean (median): {jobskills[f'AI_{end_year}_median'].mean():.3f}")
    print(f"      Mean (p05): {jobskills[f'AI_{end_year}_p05'].mean():.3f}")
    print(f"      Mean (p95): {jobskills[f'AI_{end_year}_p95'].mean():.3f}")
    
    # Sample rows for verification
    print("\n📋 Sample projections:")
    sample_cols = ['Skill', 'AI_Score_2024', f'AI_{end_year}_p05', 
                   f'AI_{end_year}_median', f'AI_{end_year}_p95']
    print(jobskills[sample_cols].head(5).to_string(index=False))
    
    print("\n✅ Projection complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()