import pandas as pd
import os


def engineer_features(input_csv, output_csv):
    print("Loading raw data...")
    df = pd.read_csv(input_csv)
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
    df['FinalPosition'] = pd.to_numeric(df['FinalPosition'], errors='coerce')
    df = df.dropna(subset=['GridPosition', 'FinalPosition'])

    # --- UPGRADE 1: EWMA of Points ---
    # 1. Map finish positions to actual F1 points
    points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    df['Points'] = df['FinalPosition'].map(points_map).fillna(0)

    # 2. Calculate Exponential Moving Average of points (span=3 races)
    # .shift(1) prevents target leakage (using today's race to predict today's race)
    print("Calculating Exponential Momentum...")
    df['Momentum_Score'] = (
        df.groupby('Abbreviation')['Points']
        .transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
    )
    df['Momentum_Score'] = df['Momentum_Score'].fillna(0.0)

    # --- UPGRADE 2: Racecraft Delta ---
    # 1. Calculate how many places they gained/lost in previous races
    df['Positions_Gained'] = df['GridPosition'] - df['FinalPosition']
    
    # 2. Get the rolling average of their overtakes
    print("Calculating Racecraft Overtake Delta...")
    df['Racecraft_Rating'] = (
        df.groupby('Abbreviation')['Positions_Gained']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df['Racecraft_Rating'] = df['Racecraft_Rating'].fillna(0.0)

    # --- FINALIZE ---
    # We now feed the AI 3 highly accurate features instead of just 2
    final_dataset = df[['Abbreviation', 'RaceName', 'GridPosition', 'Momentum_Score', 'Racecraft_Rating', 'FinalPosition']]

    final_dataset.to_csv(output_csv, index=False)
    print(f"\nAdvanced feature engineering complete. Saved to {output_csv}")


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
