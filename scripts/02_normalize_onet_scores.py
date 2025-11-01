import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 

# --- DATA LOADING ---
# Load your Step 1 output
# NOTE: Using the direct path provided in the previous turn for consistency
master_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_occupations_master.csv")

# --- DATA CLEANING & IMPUTATION (The fix for the new ValueError) ---
columns_to_process = ['Importance', 'Level']

# 1. Replace 'Not relevant' with 0, as requested
print("Replacing 'Not relevant' with 0...")
master_df[columns_to_process] = master_df[columns_to_process].replace('Not relevant', 0)

# 2. Replace 'Not available' with NaN for later imputation (as handled previously)
print("Replacing 'Not available' with NaN...")
master_df[columns_to_process] = master_df[columns_to_process].replace('Not available', np.nan)


# 3. Ensure the columns are numeric (float) after replacements
# This step should now succeed since only numbers and NaNs remain.
print("Converting columns to float...")
master_df[columns_to_process] = master_df[columns_to_process].astype(float)

# 4. Impute the remaining NaN values (those that were 'Not available') with the mean
print("Imputing NaN values with the mean...")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
master_df[columns_to_process] = imputer.fit_transform(master_df[columns_to_process])


# --- NORMALIZATION & SCORING ---
print("Applying normalization and calculating SkillDemand...")
# Normalize Importance and Level between 0 and 1
scaler = MinMaxScaler()
master_df[['Importance_norm', 'Level_norm']] = scaler.fit_transform(master_df[columns_to_process])

# Weighted Skill Demand Score
master_df['SkillDemand'] = 0.6 * master_df['Importance_norm'] + 0.4 * master_df['Level_norm']

# Save this as the processed version
master_df.to_csv("skills_occupations_normalized.csv", index=False)

# Assuming 'master_df' is the DataFrame after step 4 of the previous script
# (After cleaning 'Not relevant', 'Not available', and imputation)

print("\nProcessing complete. First 5 rows of the resulting DataFrame:")
print(master_df.head())