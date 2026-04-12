import pandas as pd
import numpy as np
import os


def engineer_features(input_csv, output_csv):
    print("Loading raw data...")
    df = pd.read_csv(input_csv)
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['FinalPosition'] = pd.to_numeric(df['FinalPosition'], errors='coerce')
    df = df.dropna(subset=['GridPosition', 'FinalPosition'])

    # Ensure chronological order within each year before computing rolling features
    if 'Round' in df.columns:
        df = df.sort_values(['Year', 'Round', 'FinalPosition']).reset_index(drop=True)
    else:
        df = df.sort_values(['Year', 'RaceName', 'FinalPosition']).reset_index(drop=True)

    # --- FEATURE 1: EWMA of Points (Momentum) ---
    # Maps finish positions to actual F1 points system
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['Points'] = df['FinalPosition'].map(points_map).fillna(0)

    # .shift(1) prevents target leakage (can't use current race to predict current race)
    print("Calculating Exponential Momentum...")
    df['Momentum_Score'] = (
        df.groupby('Abbreviation')['Points']
        .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
    )
    df['Momentum_Score'] = df['Momentum_Score'].fillna(0.0)

    # --- FEATURE 2: Racecraft Delta ---
    # How many positions they typically gain/lose during races
    df['Positions_Gained'] = df['GridPosition'] - df['FinalPosition']

    print("Calculating Racecraft Overtake Delta...")
    df['Racecraft_Rating'] = (
        df.groupby('Abbreviation')['Positions_Gained']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df['Racecraft_Rating'] = df['Racecraft_Rating'].fillna(0.0)

    # --- FEATURE 3: Constructor Strength ---
    # Team-level EWMA of combined driver points — captures overall car competitiveness
    print("Calculating Constructor Strength...")
    if 'TeamName' in df.columns:
        # Sum points per team per race, then assign back to each driver
        race_id_cols = ['Year', 'RaceName'] if 'RaceName' in df.columns else ['Year']
        team_race_points = df.groupby(race_id_cols + ['TeamName'])['Points'].sum().reset_index()
        team_race_points = team_race_points.rename(columns={'Points': 'TeamPoints'})

        # Compute EWMA of team points (shifted to avoid leakage)
        team_race_points = team_race_points.sort_values(race_id_cols)
        team_race_points['Constructor_Strength'] = (
            team_race_points.groupby('TeamName')['TeamPoints']
            .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
        )
        team_race_points['Constructor_Strength'] = team_race_points['Constructor_Strength'].fillna(0.0)

        # Merge back to driver-level data
        df = df.merge(team_race_points[race_id_cols + ['TeamName', 'Constructor_Strength']],
                       on=race_id_cols + ['TeamName'], how='left')
        df['Constructor_Strength'] = df['Constructor_Strength'].fillna(0.0)
    else:
        df['Constructor_Strength'] = 0.0

    # --- FEATURE 4: Consistency ---
    # Rolling standard deviation of finish positions — lower = more consistent
    print("Calculating Driver Consistency...")
    df['Consistency'] = (
        df.groupby('Abbreviation')['FinalPosition']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
    )
    # Fill NaN (first race or single-race window) with a neutral value
    # Use the overall median consistency as a reasonable default
    median_consistency = df['Consistency'].median()
    df['Consistency'] = df['Consistency'].fillna(median_consistency if not pd.isna(median_consistency) else 2.89)

    # --- FINALIZE ---
    # Keep metadata columns for prepare_grid.py and analysis, plus all 5 features + target
    keep_cols = ['Abbreviation', 'RaceName', 'Year', 'TeamName',
                 'GridPosition', 'Momentum_Score', 'Racecraft_Rating',
                 'Constructor_Strength', 'Consistency', 'FinalPosition']
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    final_dataset = df[keep_cols]

    final_dataset.to_csv(output_csv, index=False)
    print(f"\nAdvanced feature engineering complete. Saved to {output_csv}")
    print(f"  Features: GridPosition, Momentum_Score, Racecraft_Rating, Constructor_Strength, Consistency")
    print(f"  Rows: {len(final_dataset)}")


def run(force=False):
    """Run feature engineering. Skips if output already exists unless force=True."""
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/f1_raw_data_master.csv")
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/f1_engineered_data.csv")

    if os.path.exists(output_csv) and not force:
        if os.path.exists(input_csv):
            input_mtime = os.path.getmtime(input_csv)
            output_mtime = os.path.getmtime(output_csv)
            if output_mtime >= input_mtime:
                print(f"Engineered data is up to date. Skipping. Use --force to re-process.")
                return
    engineer_features(input_csv, output_csv)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run(force=force)
