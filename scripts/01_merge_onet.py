import pandas as pd
import os

# === 1. Folder path containing all your skill CSVs ===
# Example: "data/skills" or "./skills"
folder_path = "C:\\Users\\aryah\\OneDrive\\Desktop\\Arya\\AI Forecasting Hackathon\\data\\onet_raw"

# === 2. Prepare a list to hold all dataframes ===
dfs = []

# === 3. Loop through all CSV files ===
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        skill_name = os.path.splitext(file)[0]  # e.g. "Machine Learning" from "Machine Learning.csv"
        file_path = os.path.join(folder_path, file)

        # Read the CSV
        df = pd.read_csv(file_path)

        # Add a column for the skill name
        df["Skill"] = skill_name

        # Append to list
        dfs.append(df)

# === 4. Concatenate everything into one dataframe ===
master_df = pd.concat(dfs, ignore_index=True)

# === 5. (Optional) Clean up columns ===
master_df.columns = master_df.columns.str.strip()  # remove spaces if any

# === 6. Quick sanity checks ===
print("Total rows:", len(master_df))
print("Unique skills:", master_df['Skill'].nunique())
print("Unique occupations:", master_df['Occupation'].nunique())

print("\nSample:")
print(master_df.head())

# === 7. Save to CSV ===
master_df.to_csv("skills_occupations_master.csv", index=False)
