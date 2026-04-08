import pandas as pd

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

if __name__ == "__main__":
    engineer_features("f1_raw_data_master.csv", "f1_engineered_data.csv")