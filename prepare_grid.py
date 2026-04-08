import pandas as pd
import json

def prepare_race_day_grid():
    print("Loading Qualifying Results...")
    with open("quali_results.json", "r") as f:
        quali_data = json.load(f)

    df_history = pd.read_csv("f1_engineered_data.csv")

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

    with open("starting_grid.json", "w") as f:
        json.dump(final_grid, f, indent=4)
    
    print("Auto-calculation complete! 'starting_grid.json' is ready for the C++ engine.")

if __name__ == "__main__":
    prepare_race_day_grid()