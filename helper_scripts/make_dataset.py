import pandas as pd
import os

atp_matches = []

for year in range(2010, 2025):
    file_name = f"atp_matches_{year}.csv"
    path = os.path.join("datasets/tennis_atps", file_name)
    
    df = pd.read_csv(path)
    atp_matches.append(df)
    
    
full_df = pd.concat(atp_matches)
full_df.to_csv("all_atp_2010_2024.csv", index=False)
