import streamlit as st
import subprocess
import pandas as pd
import json
import os

# --- ABSOLUTE PATH CONFIGURATION ---
# This ensures the app always finds the right files, regardless of where the terminal is open.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "../models")

QUALI_FILE = os.path.join(DATA_DIR, "quali_results.json")
STARTING_GRID_FILE = os.path.join(DATA_DIR, "starting_grid.json")
PREPARE_SCRIPT = os.path.join(SCRIPT_DIR, "prepare_grid.py")

# On Mac/Linux, this would just be "inference", but on Windows it's "inference.exe"
INFERENCE_EXE = os.path.join(MODELS_DIR, "inference.exe")
MODEL_METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")


# --- UI Configuration ---
st.set_page_config(page_title="F1 Grand Prix Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ AI Grand Prix Simulator")
st.markdown("""
This engine automatically calculates driver momentum from historical data, requiring only 
Qualifying results to run a full 20-car race simulation via a C++ backend.
""")
st.divider()

# --- 1. Load and Display the Qualifying Grid ---
st.subheader("📊 Weekend Qualifying Results")
try:
    with open(QUALI_FILE, "r") as f:
        quali_data = json.load(f)
    
    df_quali = pd.DataFrame(quali_data)
    df_quali.columns = ["Driver", "Grid Position"]
    st.dataframe(df_quali, width='stretch', hide_index=True)
    
except FileNotFoundError:
    st.error(f"Could not find {QUALI_FILE}. Please ensure the file exists in your data folder.")

st.divider()

# --- 2. Execute the Simulation ---
if st.button("Run Full Simulation", type="primary", width='stretch'):
    with st.spinner("Calculating Historical Form and Executing C++ Engine..."):
        try:
            # Step A: Run the Python Bridge Script to calculate Recent Form
            # We pass the absolute path to ensure it runs correctly
            subprocess.run(["python", PREPARE_SCRIPT], check=True)
            
            # Check if required files exist before running C++
            if not os.path.exists(STARTING_GRID_FILE):
                st.error("Error: starting_grid.json was not created. Check prepare_grid.py output.")
                st.stop()
            if not os.path.exists(MODEL_METADATA_FILE):
                st.error("Error: model_metadata.json not found. Please run model training first.")
                st.stop()
            if not os.path.exists(INFERENCE_EXE):
                st.error("Error: inference.exe not found. Please compile the C++ engine.")
                st.stop()
            
            # Step B: Run the C++ Inference Engine
            # Added encoding="utf-8" to prevent Windows emoji crashes
            result = subprocess.run([INFERENCE_EXE], capture_output=True, text=True, encoding="utf-8", check=True)
            
            st.success("Simulation Complete!")
            
            # Display the final leaderboard exactly as C++ formatted it
            st.code(result.stdout, language="text")
            
            # (Optional) Expandable section to show the professor the math behind it
            with open(STARTING_GRID_FILE, "r") as f:
                auto_grid = pd.DataFrame(json.load(f))
                auto_grid.columns = ["Driver", "Grid Position", "Auto-Calculated Form"]
            
            with st.expander("View Auto-Calculated Input Math"):
                st.markdown("These are the final variables Python sent to the C++ Engine.")
                st.dataframe(auto_grid, width='stretch', hide_index=True)
            
        except FileNotFoundError:
            st.error(f"Engine Error: Could not find '{INFERENCE_EXE}'. Ensure you compiled it inside the models/ folder.")
        except subprocess.CalledProcessError as e:
             st.error(f"Execution Error. The underlying script failed: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")