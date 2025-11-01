import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 

master_df = pd.read_csv("C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\skills_occupations_master.csv")
columns_to_process = ['Importance', 'Level']

master_df[columns_to_process] = master_df[columns_to_process].replace('Not relevant', 0)

master_df[columns_to_process] = master_df[columns_to_process].replace('Not available', np.nan)

master_df[columns_to_process] = master_df[columns_to_process].astype(float)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
master_df[columns_to_process] = imputer.fit_transform(master_df[columns_to_process])

scaler = MinMaxScaler()
master_df[['Importance_norm', 'Level_norm']] = scaler.fit_transform(master_df[columns_to_process])

master_df['SkillDemand'] = 0.6 * master_df['Importance_norm'] + 0.4 * master_df['Level_norm']
master_df.to_csv("skills_occupations_normalized.csv", index=False)