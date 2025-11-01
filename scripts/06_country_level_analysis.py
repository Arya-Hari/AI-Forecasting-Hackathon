import pandas as pd
import numpy as np

CONFIG = {
    'occupation_impact_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\occupation_ai_impact_final.csv",
    'capacity_indicators_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\country_capacity_indicators.csv",
    'output_path': "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\occupation_impact_by_country.csv",
    'years': list(range(2024, 2036))
}

def calculate_capacity_score(country_data):
    indicator_mapping = {
        'Public R&D expenditure (% of GDP)': ('rd_expenditure', 2.0),
        'Electricity Access (% of population)': ('electricity', 100.0),
        'IT service exports (% of GDP)': ('it_exports', 50.0),
        'Internet users (% of population)': ('internet', 100.0),
        'Fixed broadband subscriptions (per 100 people)': ('broadband', 40.0),
        'High-tech exports (% of manufactured exports)': ('hightech_exports', 30.0),
        'Government Effectiveness (WGI, estimate)': ('govt_effectiveness', 5.0, True),
        'Researchers in R&D (per million people)': ('researchers', 8000.0)
    }
    
    indicators = {}
    
    for col_name, (key, max_val, *args) in indicator_mapping.items():
        try:
            value = pd.to_numeric(country_data[col_name], errors='coerce')
            if pd.isna(value):
                indicators[key] = 0.0
                continue
                
            if args and args[0]:  
                value = (value + 2.5) / max_val
            else:
                value = value / max_val
            indicators[key] = np.clip(value, 0.0, 1.0)
        except (KeyError, ValueError, TypeError):
            indicators[key] = 0.0
    
    weights = {
        'rd_expenditure': 0.15,
        'electricity': 0.10,
        'it_exports': 0.15,
        'internet': 0.15,
        'broadband': 0.10,
        'hightech_exports': 0.10,
        'govt_effectiveness': 0.15,
        'researchers': 0.10
    }
    
    capacity_score = sum(indicators[k] * weights[k] for k in weights)
    return capacity_score

def calculate_adoption_multiplier(capacity_score):
    return 0.3 + 1.2 / (1 + np.exp(-5 * (capacity_score - 0.5)))

def main():
    print("=" * 60)
    print("COUNTRY-SPECIFIC AI IMPACT ANALYSIS")
    print("=" * 60)
    
    print("\n📂 Loading data...")
    occupation_df = pd.read_csv(CONFIG['occupation_impact_path'])
    capacity_df = pd.read_csv(CONFIG['capacity_indicators_path'])
    
    country_columns = [col for col in capacity_df.columns if 'Formula' in col]
    countries = [col.replace(' (Formula)', '') for col in country_columns]
    
    country_data_list = []
    
    for country_col, country_name in zip(country_columns, countries):
        country_series = capacity_df[['Variable', country_col]].copy()
        country_series.columns = ['Variable', 'Value']
        country_series['Country'] = country_name
        country_series['Value'] = pd.to_numeric(country_series['Value'], errors='coerce')
        country_data_list.append(country_series)
    
    capacity_long = pd.concat(country_data_list, ignore_index=True)
    
    capacity_df = capacity_long.pivot(index='Country', columns='Variable', values='Value').reset_index()
    
    print(f"   ✅ Occupations: {len(occupation_df)}")
    print(f"   ✅ Countries: {len(capacity_df)}")
    
    print("\n🔧 Calculating capacity scores...")
    capacity_df['Capacity_Score'] = capacity_df.apply(calculate_capacity_score, axis=1)
    capacity_df['Adoption_Multiplier'] = capacity_df['Capacity_Score'].apply(calculate_adoption_multiplier)
    
    print("\n📊 Capacity scores:")
    print(capacity_df[['Country', 'Capacity_Score', 'Adoption_Multiplier']].to_string(index=False))
    
    print("\n🔗 Creating country-occupation combinations...")
    results = []
    
    for _, country in capacity_df.iterrows():
        country_name = country['Country']
        multiplier = country['Adoption_Multiplier']
        
        country_occupations = occupation_df.copy()
        country_occupations['Country'] = country_name
        country_occupations['Capacity_Score'] = country['Capacity_Score']
        country_occupations['Adoption_Multiplier'] = multiplier
        
        for year in CONFIG['years']:
            years_elapsed = year - 2024
            
            effective_year = 2024 + (years_elapsed * multiplier)
            effective_year_int = min(2035, int(np.round(effective_year)))
            
            country_occupations[f'Adjusted_Vulnerability_Median_{year}'] = \
                country_occupations[f'Vulnerability_Median_{effective_year_int}']
            country_occupations[f'Adjusted_Vulnerability_p05_{year}'] = \
                country_occupations[f'Vulnerability_p05_{effective_year_int}']
            country_occupations[f'Adjusted_Vulnerability_p95_{year}'] = \
                country_occupations[f'Vulnerability_p95_{effective_year_int}']
        
        results.append(country_occupations)
    
    final_df = pd.concat(results, ignore_index=True)
    
    print(f"   ✅ Created {len(final_df):,} country-occupation combinations")
    
    print(f"\n💾 Saving to {CONFIG['output_path']}...")
    final_df.to_csv(CONFIG['output_path'], index=False)
    
    print("\n📈 Sample results:")
    sample_cols = ['Country', 'Occupation', 'Capacity_Score', 
                   'Adjusted_Vulnerability_Median_2024', 'Adjusted_Vulnerability_Median_2035']
    print(final_df[sample_cols].head(10).to_string(index=False))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()