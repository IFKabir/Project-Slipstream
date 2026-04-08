import pandas as pd
import json
import os

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

QUALI_FILE = os.path.join(DATA_DIR, "quali_results.json")
ENGINEERED_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "starting_grid.json")


def prepare_race_day_grid():
    print("Loading Qualifying Results...")
    try:
        with open(QUALI_FILE, "r") as f:
            quali_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {QUALI_FILE} not found.")
        return

    try:
        df_history = pd.read_csv(ENGINEERED_FILE)
    except FileNotFoundError:
        print(f"Error: {ENGINEERED_FILE} not found.")
        return

    final_grid = []

    points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

    for entry in quali_data:
        driver = entry["driver"]
        grid_pos = entry["grid_pos"]

        driver_history = df_history[df_history["Abbreviation"] == driver]

        if len(driver_history) > 0:
            recent_races = driver_history.tail(3)
            momentum_score = recent_races["FinalPosition"].map(points_map).fillna(0).ewm(span=3, adjust=False).mean().iloc[-1]
            racecraft_rating = float(recent_races["GridPosition"].iloc[-1] - recent_races["FinalPosition"].iloc[-1])
        else:
            momentum_score = 0.0
            racecraft_rating = 0.0

        final_grid.append({
            "driver": driver,
            "grid_pos": float(grid_pos),
            "momentum_score": float(momentum_score),
            "racecraft_rating": float(racecraft_rating)
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_grid, f, indent=4)

    print("Auto-calculation complete! 'starting_grid.json' is ready for the C++ engine.")


if __name__ == "__main__":
    prepare_race_day_grid()