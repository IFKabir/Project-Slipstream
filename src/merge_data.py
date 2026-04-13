"""Merge all per-year f1_raw_*.csv files into f1_raw_data_master.csv."""
import pandas as pd
import glob
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

files = sorted(glob.glob(os.path.join(DATA_DIR, "f1_raw_20*.csv")))

print("Files found:")
for f in files:
    print(f"  {os.path.basename(f)}")

dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates()
combined = combined.sort_values(["Year", "RaceName", "GridPosition"]).reset_index(drop=True)

output = os.path.join(DATA_DIR, "f1_raw_data_master.csv")
combined.to_csv(output, index=False)

years = sorted(combined["Year"].unique())
print(f"\nMerged {len(files)} files -> {len(combined)} rows")
print(f"Years: {years}")
print(f"Saved to: {output}")
