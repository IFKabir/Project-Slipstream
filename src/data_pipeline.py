import fastf1
import pandas as pd
import os
import sys

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "f1_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "f1_raw_data_master.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_season_data(year):
    print(f"Fetching data for the {year} season...")
    schedule = fastf1.get_event_schedule(year)
    season_data = []

    for index, event in schedule.iterrows():
        race_name = event['EventName']
        print(f"Processing: {race_name}")

        try:
            qualifying = fastf1.get_session(year, race_name, 'Q')
            race = fastf1.get_session(year, race_name, 'R')

            qualifying.load(telemetry=False, weather=False)
            race.load(telemetry=False, weather=False)

            q_results = qualifying.results[['Abbreviation', 'Position']].rename(columns={'Position': 'GridPosition'})
            r_results = race.results[['Abbreviation', 'Position']].rename(columns={'Position': 'FinalPosition'})

            merged_data = pd.merge(q_results, r_results, on='Abbreviation')
            merged_data['RaceName'] = race_name
            merged_data['Year'] = year

            season_data.append(merged_data)

        except Exception as e:
            print(f"Could not load data for {race_name}: {e}")

    return pd.concat(season_data, ignore_index=True)


def run(force=False):
    """Run the data pipeline. Skips if data already exists unless force=True."""
    if os.path.exists(OUTPUT_FILE) and not force:
        print(f"Data already exists at {OUTPUT_FILE}. Skipping fetch. Use --force to re-download.")
        return

    # Ergast API and F1 live timing changes broke automatic fetches for >= 2025.
    # Use 2024 as the stable training dataset to keep exactly 1 year of data.
    current_year = 2026
    df_current = fetch_season_data(current_year)

    df_current.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMaster Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)