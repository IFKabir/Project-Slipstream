import pandas as pd
import json

def prepare_race_day_grid():
    print("Loading Qualifying Results...")
    # 1. Load the simple user input
    with open("quali_results.json", "r") as f:
        quali_data = json.load(f)

    # 2. Load the historical database to calculate form
    df_history = pd.read_csv("f1_engineered_data.csv")

    final_grid = []

    for entry in quali_data:
        driver = entry["driver"]
        grid_pos = entry["grid_pos"]

        # Find the driver's historical races
        driver_history = df_history[df_history["Abbreviation"] == driver]

        # Calculate their recent form (average of last 3 finishes)
        if len(driver_history) > 0:
            # Get the last 3 rows (most recent races)
            recent_races = driver_history.tail(3)
            recent_form = recent_races["FinalPosition"].mean()
        else:
            # If it's a brand new rookie with no data, give them a neutral score of 10
            recent_form = 10.0

        # Append the calculated data to the final grid
        final_grid.append({
            "driver": driver,
            "grid_pos": float(grid_pos),
            "recent_form": float(recent_form)
        })

    # 3. Save the complete data for the C++ Engine
    with open("starting_grid.json", "w") as f:
        json.dump(final_grid, f, indent=4)
    
    print("Auto-calculation complete! 'starting_grid.json' is ready for the C++ engine.")

if __name__ == "__main__":
    prepare_race_day_grid()