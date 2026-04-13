import pandas as pd
import numpy as np
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

QUALI_FILE = os.path.join(DATA_DIR, "quali_results.json")
ENGINEERED_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "starting_grid.json")


def prepare_race_day_grid():
    """
    Prepare the starting grid for race-day inference.
    
    Computes all 5 model features for each driver based on their historical data,
    exactly matching the feature computation used during training to avoid
    train/inference skew.
    """
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

    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

    final_grid = []

    for entry in quali_data:
        driver = entry["driver"]
        grid_pos = entry["grid_pos"]

        driver_history = df_history[df_history["Abbreviation"] == driver]

        if len(driver_history) > 0:
            recent_races = driver_history.tail(3)

            # --- Feature 1: Momentum Score ---
            recent_points = recent_races["FinalPosition"].map(points_map).fillna(0)
            momentum_score = float(recent_points.ewm(span=3, adjust=False).mean().iloc[-1])

            # --- Feature 2: Racecraft Rating ---
            positions_gained = recent_races["GridPosition"] - recent_races["FinalPosition"]
            racecraft_rating = float(positions_gained.mean())

            # --- Feature 3: Constructor Strength ---
            if "Constructor_Strength" in driver_history.columns:
                constructor_strength = float(driver_history["Constructor_Strength"].iloc[-1])
            else:
                if "TeamName" in driver_history.columns:
                    team = driver_history["TeamName"].iloc[-1]
                    team_drivers = df_history[df_history["TeamName"] == team]
                    team_points = team_drivers.tail(6)["FinalPosition"].map(points_map).fillna(0)
                    constructor_strength = float(team_points.ewm(span=3, adjust=False).mean().iloc[-1])
                else:
                    constructor_strength = 0.0

            # --- Feature 4: Consistency ---
            if "Consistency" in driver_history.columns and not pd.isna(driver_history["Consistency"].iloc[-1]):
                consistency = float(driver_history["Consistency"].iloc[-1])
            else:
                consistency = float(recent_races["FinalPosition"].std()) if len(recent_races) > 1 else 2.89
        else:
            momentum_score = 0.0
            racecraft_rating = 0.0
            constructor_strength = 0.0
            consistency = 2.89

        final_grid.append({
            "driver": driver,
            "GridPosition": float(grid_pos),
            "Momentum_Score": round(momentum_score, 4),
            "Racecraft_Rating": round(racecraft_rating, 4),
            "Constructor_Strength": round(constructor_strength, 4),
            "Consistency": round(consistency, 4)
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_grid, f, indent=4)

    print(f"Auto-calculation complete! {len(final_grid)} drivers prepared.")
    print(f"  Features: GridPosition, Momentum_Score, Racecraft_Rating, Constructor_Strength, Consistency")
    print(f"  Output: '{OUTPUT_FILE}' is ready for the C++ engine.")


if __name__ == "__main__":
    prepare_race_day_grid()