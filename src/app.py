import streamlit as st
import subprocess
import pandas as pd
import json

st.set_page_config(page_title="F1 Grand Prix Predictor", page_icon="🏎️")

st.title("🏎️ AI Grand Prix Simulator")
st.markdown("This engine automatically calculates driver form from historical data, requiring only Qualifying results to run a full race simulation.")
st.divider()

st.subheader("📊 Weekend Qualifying Results")
try:
    with open("quali_results.json", "r") as f:
        quali_data = json.load(f)
    
    df_quali = pd.DataFrame(quali_data)
    df_quali.columns = ["Driver", "Grid Position"]
    st.dataframe(df_quali, width='stretch', hide_index=True)
    
except FileNotFoundError:
    st.error("Could not find 'quali_results.json'.")

st.divider()

if st.button("Run Full Simulation", type="primary", width='stretch'):
    with st.spinner("Calculating Historical Form and Executing C++ Engine..."):
        try:
            subprocess.run(["python", "prepare_grid.py"], check=True)
            
            result = subprocess.run(["../models/inference.exe"], capture_output=True, text=True, encoding="utf-8", check=True)
            
            st.success("Simulation Complete!")
            st.code(result.stdout, language="text")
            
            with open("../data/starting_grid.json", "r") as f:
                auto_grid = pd.DataFrame(json.load(f))
                auto_grid.columns = ["Driver", "Grid Position", "Auto-Calculated Form"]
            with st.expander("View Auto-Calculated Math"):
                st.dataframe(auto_grid, width='stretch', hide_index=True)
            
        except Exception as e:
            st.error(f"Error: {e}")