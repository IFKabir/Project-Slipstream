import pandas as pd
import os
import sys

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_DIR, "f1_raw_data_master.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")


def engineer_features(input_csv, output_csv):
    print("Loading raw data...")
    df = pd.read_csv(input_csv)

    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')

    print("Calculating Driver Form (Rolling Average)...")
    df['Recent_3_Race_Avg'] = (
        df.groupby('Abbreviation')['FinalPosition']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df['Recent_3_Race_Avg'] = df['Recent_3_Race_Avg'].fillna(10.0)

    df = df.dropna(subset=['GridPosition', 'FinalPosition'])

    final_dataset = df[['Abbreviation', 'RaceName', 'GridPosition', 'Recent_3_Race_Avg', 'FinalPosition']]

    final_dataset.to_csv(output_csv, index=False)
    print(f"\nFeature engineering complete. Saved to {output_csv}")


def run(force=False):
    """Run feature engineering. Skips if output already exists unless force=True."""
    if os.path.exists(OUTPUT_FILE) and not force:
        # Check if output is newer than input (i.e. already up to date)
        if os.path.exists(INPUT_FILE):
            input_mtime = os.path.getmtime(INPUT_FILE)
            output_mtime = os.path.getmtime(OUTPUT_FILE)
            if output_mtime >= input_mtime:
                print(f"Engineered data is up to date. Skipping. Use --force to re-process.")
                return
    engineer_features(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)