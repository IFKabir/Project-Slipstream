import fastf1
import fastf1.exceptions
import pandas as pd
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "f1_cache")
OUTPUT_FILE = os.path.join(DATA_DIR, "f1_raw_data_master.csv")

# Rate-limit settings
REQUEST_DELAY = 8          # seconds between session loads (keeps us well under 500/h)
RATE_LIMIT_WAIT = 3600     # seconds to wait if rate-limited (1 hour)
MAX_RETRIES = 3            # max retries per session on rate-limit

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def _load_with_retry(session, retries=MAX_RETRIES):
    """Load a session with automatic retry on rate-limit errors."""
    for attempt in range(1, retries + 1):
        try:
            session.load(telemetry=False, weather=False)
            return
        except fastf1.exceptions.RateLimitExceededError:
            if attempt < retries:
                print(f"  ⚠ Rate limit hit. Waiting {RATE_LIMIT_WAIT}s before retry "
                      f"({attempt}/{retries})...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                raise


def fetch_season_data(year):
    print(f"\nFetching data for the {year} season...")
    schedule = fastf1.get_event_schedule(year)
    season_data = []

    for index, event in schedule.iterrows():
        race_name = event['EventName']
        print(f"Processing: {race_name}")

        try:
            qualifying = fastf1.get_session(year, race_name, 'Q')
            race = fastf1.get_session(year, race_name, 'R')

            _load_with_retry(qualifying)
            time.sleep(REQUEST_DELAY)

            _load_with_retry(race)
            time.sleep(REQUEST_DELAY)

            q_results = qualifying.results[['Abbreviation', 'Position', 'TeamName']].rename(columns={'Position': 'GridPosition'})
            r_results = race.results[['Abbreviation', 'Position', 'TeamName']].rename(columns={'Position': 'FinalPosition'})

            merged_data = pd.merge(q_results, r_results[['Abbreviation', 'FinalPosition']], on='Abbreviation')
            merged_data['RaceName'] = race_name
            merged_data['Year'] = year

            season_data.append(merged_data)

        except fastf1.exceptions.RateLimitExceededError:
            print(f"  ✗ Rate limit exceeded for {race_name} after {MAX_RETRIES} retries. Skipping.")
        except Exception as e:
            print(f"  ✗ Could not load data for {race_name}: {e}")

    if season_data:
        return pd.concat(season_data, ignore_index=True)
    return pd.DataFrame()


def _year_cache_path(year):
    """Path for per-year incremental cache file."""
    return os.path.join(DATA_DIR, f"f1_raw_{year}.csv")


def run(force=False):
    """Run the data pipeline. Skips if data already exists unless force=True."""
    if os.path.exists(OUTPUT_FILE) and not force:
        print(f"Data already exists at {OUTPUT_FILE}. Skipping fetch. Use --force to re-download.")
        return

    target_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    all_seasons_data = []

    for year in target_years:
        year_file = _year_cache_path(year)

        if os.path.exists(year_file) and not force:
            print(f"\n✓ Using cached data for {year} ({year_file})")
            df_season = pd.read_csv(year_file)
        else:
            df_season = fetch_season_data(year)
            if not df_season.empty:
                df_season.to_csv(year_file, index=False)
                print(f"  ✓ Saved {year} data to {year_file}")

        if not df_season.empty:
            all_seasons_data.append(df_season)

    if all_seasons_data:
        df_combined = pd.concat(all_seasons_data, ignore_index=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)
        print(f"\nMaster Data saved to {OUTPUT_FILE}")
        print(f"Total historical records downloaded: {len(df_combined)}")
    else:
        print("\n✗ No data was fetched.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)