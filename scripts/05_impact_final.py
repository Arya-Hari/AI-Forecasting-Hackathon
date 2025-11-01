import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
CONFIG = {
    'projections_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_with_projections.csv",
    'output_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\occupation_ai_impact_final.csv",
    'start_year': 2024,
    'end_year': 2035,
    'annual_demand_decay': 0.005,  # 2.5% per year
    'aggregation_method': 'max',  # Options: 'max', 'mean', 'weighted_mean'
}

# --- HELPER FUNCTIONS ---

def calculate_demand_decay(initial_demand, years_elapsed, decay_rate):
    """
    Calculate skill demand with exponential decay.
    
    Formula: Demand_t = Demand_0 * (1 - decay_rate)^t
    """
    return initial_demand * np.power(1 - decay_rate, years_elapsed)

def aggregate_vulnerability(group_df, method='max'):
    """
    Aggregate skill-level vulnerability to occupation level.
    
    Methods:
    - 'max': Highest vulnerability skill (bottleneck approach)
    - 'mean': Average vulnerability across skills
    - 'weighted_mean': Weighted by original SkillDemand
    """
    vuln_cols = [c for c in group_df.columns if c.startswith('Vulnerability_')]
    
    if method == 'max':
        return group_df[vuln_cols].max()
    elif method == 'mean':
        return group_df[vuln_cols].mean()
    elif method == 'weighted_mean':
        weights = group_df['SkillDemand']
        weighted_avg = {}
        for col in vuln_cols:
            weighted_avg[col] = np.average(group_df[col], weights=weights)
        return pd.Series(weighted_avg)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

# --- MAIN PROCESSING ---

def main():
    print("=" * 60)
    print("OCCUPATION AI IMPACT ANALYSIS")
    print("=" * 60)
    
    # --- 1. LOAD DATA ---
    print(f"\n📂 Loading projections from {CONFIG['projections_path']}...")
    try:
        proj_df = pd.read_csv(CONFIG['projections_path'])
        print(f"   ✅ Loaded {len(proj_df):,} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"Projections file not found: {CONFIG['projections_path']}")
    
    # --- 2. VALIDATE DATA ---
    print("\n🔍 Validating data structure...")
    required_cols = ['Occupation', 'Code', 'Skill', 'SkillDemand']
    missing_cols = [c for c in required_cols if c not in proj_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    years = list(range(CONFIG['start_year'], CONFIG['end_year'] + 1))
    print(f"   Years to process: {len(years)} ({CONFIG['start_year']}-{CONFIG['end_year']})")
    
    # Check for required AI projection columns
    missing_years = []
    for year in years:
        if f'AI_{year}_median' not in proj_df.columns:
            missing_years.append(year)
    
    if missing_years:
        raise ValueError(f"Missing AI projection columns for years: {missing_years}")
    
    print("   ✅ All required columns present")
    
    # --- 3. INITIAL STATISTICS ---
    print("\n📊 Initial data summary:")
    print(f"   Unique occupations: {proj_df['Occupation'].nunique()}")
    print(f"   Unique skills: {proj_df['Skill'].nunique()}")
    print(f"   SkillDemand range: [{proj_df['SkillDemand'].min():.3f}, {proj_df['SkillDemand'].max():.3f}]")
    print(f"   SkillDemand mean: {proj_df['SkillDemand'].mean():.3f}")
    
    # --- 4. CALCULATE DYNAMIC SKILL DEMAND & VULNERABILITY ---
    print(f"\n🔧 Calculating vulnerability scores with {CONFIG['annual_demand_decay']*100:.1f}% annual demand decay...")
    
    # Store baseline demand
    proj_df['SkillDemand_2024'] = proj_df['SkillDemand']
    
    for year in years:
        delta_years = year - CONFIG['start_year']
        
        # Calculate decaying skill demand
        demand_col = f'SkillDemand_{year}'
        proj_df[demand_col] = calculate_demand_decay(
            proj_df['SkillDemand_2024'],
            delta_years,
            CONFIG['annual_demand_decay']
        )
        proj_df[demand_col] = proj_df[demand_col].clip(0.0, 1.0)
        
        # Calculate vulnerability for each confidence interval
        # Vulnerability = Decaying Demand × AI Capability Score
        
        # Median projection
        proj_df[f'Vulnerability_Median_{year}'] = (
            proj_df[demand_col] * proj_df[f'AI_{year}_median']
        )
        
        # 5th percentile (optimistic - lower AI capability)
        proj_df[f'Vulnerability_p05_{year}'] = (
            proj_df[demand_col] * proj_df[f'AI_{year}_p05']
        )
        
        # 95th percentile (pessimistic - higher AI capability)
        proj_df[f'Vulnerability_p95_{year}'] = (
            proj_df[demand_col] * proj_df[f'AI_{year}_p95']
        )
    
    # --- 5. SHOW SAMPLE CALCULATIONS ---
    print("\n   Sample vulnerability calculation for 2035:")
    sample_skill = proj_df.iloc[0]
    print(f"   Skill: {sample_skill['Skill']}")
    print(f"   Initial demand (2024): {sample_skill['SkillDemand_2024']:.3f}")
    print(f"   Decayed demand (2035): {sample_skill['SkillDemand_2035']:.3f}")
    print(f"   AI capability (median, 2035): {sample_skill['AI_2035_median']:.3f}")
    print(f"   Vulnerability (2035): {sample_skill['Vulnerability_Median_2035']:.3f}")
    
    # --- 6. AGGREGATE TO OCCUPATION LEVEL ---
    print(f"\n📊 Aggregating to occupation level using '{CONFIG['aggregation_method']}' method...")
    
    vulnerability_cols = [c for c in proj_df.columns if c.startswith('Vulnerability_')]
    print(f"   Found {len(vulnerability_cols)} vulnerability columns")
    
    # Group and aggregate
    if CONFIG['aggregation_method'] == 'weighted_mean':
        occupation_impact = proj_df.groupby(['Occupation', 'Code']).apply(
            lambda g: aggregate_vulnerability(g, method='weighted_mean')
        ).reset_index()
    else:
        occupation_impact = proj_df.groupby(['Occupation', 'Code'])[vulnerability_cols].agg(
            CONFIG['aggregation_method']
        ).reset_index()
    
    print(f"   ✅ Aggregated to {len(occupation_impact):,} occupations")
    
    # --- 7. ANALYZE RESULTS ---
    print("\n📈 Vulnerability Analysis:")
    
    # 2024 baseline
    print(f"\n   2024 (baseline):")
    print(f"      Mean: {occupation_impact['Vulnerability_Median_2024'].mean():.3f}")
    print(f"      Max: {occupation_impact['Vulnerability_Median_2024'].max():.3f}")
    
    # 2035 projections
    print(f"\n   2035 (projected - median):")
    print(f"      Mean: {occupation_impact['Vulnerability_Median_2035'].mean():.3f}")
    print(f"      Max: {occupation_impact['Vulnerability_Median_2035'].max():.3f}")
    
    print(f"\n   2035 (projected - 95th percentile):")
    print(f"      Mean: {occupation_impact['Vulnerability_p95_2035'].mean():.3f}")
    print(f"      Max: {occupation_impact['Vulnerability_p95_2035'].max():.3f}")
    
    # Calculate change
    change = ((occupation_impact['Vulnerability_Median_2035'] - 
               occupation_impact['Vulnerability_Median_2024']) / 
              occupation_impact['Vulnerability_Median_2024'] * 100)
    print(f"\n   Average change (2024→2035): {change.mean():.1f}%")
    
    # --- 8. TOP VULNERABLE OCCUPATIONS ---
    print("\n🚨 Top 10 Most Vulnerable Occupations (2035, Median):")
    top_vulnerable = occupation_impact.nlargest(10, 'Vulnerability_Median_2035')
    print(top_vulnerable[['Occupation', 'Code', 'Vulnerability_Median_2024', 
                          'Vulnerability_Median_2035', 'Vulnerability_p95_2035']].to_string(index=False))
    
    # --- 9. LEAST VULNERABLE OCCUPATIONS ---
    print("\n✅ Top 10 Least Vulnerable Occupations (2035, Median):")
    least_vulnerable = occupation_impact.nsmallest(10, 'Vulnerability_Median_2035')
    print(least_vulnerable[['Occupation', 'Code', 'Vulnerability_Median_2024', 
                            'Vulnerability_Median_2035']].to_string(index=False))
    
    # --- 10. SAVE RESULTS ---
    print(f"\n💾 Saving results to {CONFIG['output_path']}...")
    occupation_impact.to_csv(CONFIG['output_path'], index=False)
    print(f"   ✅ Saved {len(occupation_impact):,} occupations with {len(occupation_impact.columns)} columns")
    
    print("\n✅ Analysis complete!")
    print("=" * 60)
    
    return occupation_impact

if __name__ == "__main__":
    result = main()