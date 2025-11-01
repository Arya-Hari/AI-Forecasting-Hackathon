import pandas as pd
import os

folder_path = "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\onet_raw"
dfs = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        skill_name = os.path.splitext(file)[0]
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["Skill"] = skill_name
        dfs.append(df)

master_df = pd.concat(dfs, ignore_index=True)
master_df.columns = master_df.columns.str.strip()
print("Total rows:", len(master_df))
print("Unique skills:", master_df['Skill'].nunique())
print("Unique occupations:", master_df['Occupation'].nunique())
print("\nSample:")
print(master_df.head())
master_df.to_csv("skills_occupations_master.csv", index=False)
