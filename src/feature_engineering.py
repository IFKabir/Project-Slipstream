import pandas as pd
import numpy as np
import os


def engineer_features(input_csv, output_csv):
    print("Loading raw data...")
    df = pd.read_csv(input_csv)
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['FinalPosition'] = pd.to_numeric(df['FinalPosition'], errors='coerce')
    df = df.dropna(subset=['GridPosition', 'FinalPosition'])

    if 'Round' in df.columns:
        df = df.sort_values(['Year', 'Round', 'FinalPosition']).reset_index(drop=True)
    else:
        df = df.sort_values(['Year', 'RaceName', 'FinalPosition']).reset_index(drop=True)

    # --- FEATURE 1: EWMA of Points (Momentum) ---
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['Points'] = df['FinalPosition'].map(points_map).fillna(0)

    print("Calculating Exponential Momentum...")
    df['Momentum_Score'] = (
        df.groupby('Abbreviation')['Points']
        .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
    )
    df['Momentum_Score'] = df['Momentum_Score'].fillna(0.0)

    # --- FEATURE 2: Racecraft Delta ---
    df['Positions_Gained'] = df['GridPosition'] - df['FinalPosition']

    print("Calculating Racecraft Overtake Delta...")
    df['Racecraft_Rating'] = (
        df.groupby('Abbreviation')['Positions_Gained']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df['Racecraft_Rating'] = df['Racecraft_Rating'].fillna(0.0)

    # --- FEATURE 3: Constructor Strength ---
    print("Calculating Constructor Strength...")
    if 'TeamName' in df.columns:
        race_id_cols = ['Year', 'RaceName'] if 'RaceName' in df.columns else ['Year']
        team_race_points = df.groupby(race_id_cols + ['TeamName'])['Points'].sum().reset_index()
        team_race_points = team_race_points.rename(columns={'Points': 'TeamPoints'})

        team_race_points = team_race_points.sort_values(race_id_cols)
        team_race_points['Constructor_Strength'] = (
            team_race_points.groupby('TeamName')['TeamPoints']
            .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
        )
        team_race_points['Constructor_Strength'] = team_race_points['Constructor_Strength'].fillna(0.0)

        df = df.merge(team_race_points[race_id_cols + ['TeamName', 'Constructor_Strength']],
                       on=race_id_cols + ['TeamName'], how='left')
        df['Constructor_Strength'] = df['Constructor_Strength'].fillna(0.0)
    else:
        df['Constructor_Strength'] = 0.0

    # --- FEATURE 4: Consistency ---
    print("Calculating Driver Consistency...")
    df['Consistency'] = (
        df.groupby('Abbreviation')['FinalPosition']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
    )
    median_consistency = df['Consistency'].median()
    df['Consistency'] = df['Consistency'].fillna(median_consistency if not pd.isna(median_consistency) else 2.89)

    # --- FEATURE 5: Teammate Grid Delta ---
    print("Calculating Teammate Grid Delta...")
    if 'TeamName' in df.columns and 'RaceName' in df.columns:
        # For each race, compute the mean grid position per team
        race_id_cols_tm = ['Year', 'RaceName', 'TeamName']
        team_avg_grid = df.groupby(race_id_cols_tm)['GridPosition'].transform('mean')
        # Delta = driver's grid position minus team average (negative = better than teammate)
        df['Teammate_Grid_Delta'] = df['GridPosition'] - team_avg_grid
        df['Teammate_Grid_Delta'] = df['Teammate_Grid_Delta'].fillna(0.0)
    else:
        df['Teammate_Grid_Delta'] = 0.0

    # --- FEATURE 6: Recent DNFs (rolling 5-race sum) ---
    print("Calculating Recent DNF Count...")
    df['Is_DNF'] = (df['FinalPosition'] >= 18).astype(int)
    df['Recent_DNFs'] = (
        df.groupby('Abbreviation')['Is_DNF']
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
    )
    df['Recent_DNFs'] = df['Recent_DNFs'].fillna(0.0)

    keep_cols = ['Abbreviation', 'RaceName', 'Year', 'TeamName',
                 'GridPosition', 'Momentum_Score', 'Racecraft_Rating',
                 'Constructor_Strength', 'Consistency',
                 'Teammate_Grid_Delta', 'Recent_DNFs',
                 'FinalPosition']
    keep_cols = [c for c in keep_cols if c in df.columns]
    final_dataset = df[keep_cols]

    final_dataset.to_csv(output_csv, index=False)
    print(f"\nAdvanced feature engineering complete. Saved to {output_csv}")
    print(f"  Features: GridPosition, Momentum_Score, Racecraft_Rating, Constructor_Strength,")
    print(f"            Consistency, Teammate_Grid_Delta, Recent_DNFs")
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
