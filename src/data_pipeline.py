import fastf1
import pandas as pd
import os

cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

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

if __name__ == "__main__":
    df_2023 = fetch_season_data(2023)
    df_2024 = fetch_season_data(2024)
    
    combined_df = pd.concat([df_2023, df_2024], ignore_index=True)
    
    combined_df.to_csv("f1_raw_data_master.csv", index=False)
    print("\nMaster Data saved to f1_raw_data_master.csv")