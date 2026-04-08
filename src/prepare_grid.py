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

    for entry in quali_data:
        driver = entry["driver"]
        grid_pos = entry["grid_pos"]

        driver_history = df_history[df_history["Abbreviation"] == driver]

        if len(driver_history) > 0:
            recent_races = driver_history.tail(3)
            recent_form = recent_races["FinalPosition"].mean()
        else:
            recent_form = 10.0

        final_grid.append({
            "driver": driver,
            "grid_pos": float(grid_pos),
            "recent_form": float(recent_form)
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_grid, f, indent=4)

    print("Auto-calculation complete! 'starting_grid.json' is ready for the C++ engine.")


if __name__ == "__main__":
    prepare_race_day_grid()