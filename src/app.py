import streamlit as st
import subprocess
import pandas as pd
import json
import sys
import os
import platform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

QUALI_FILE = os.path.join(DATA_DIR, "quali_results.json")
STARTING_GRID_FILE = os.path.join(DATA_DIR, "starting_grid.json")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

if platform.system() == "Windows":
    INFERENCE_EXE = os.path.join(MODELS_DIR, "inference.exe")
else:
    INFERENCE_EXE = os.path.join(MODELS_DIR, "inference")

MODEL_METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

sys.path.insert(0, SCRIPT_DIR)
import data_pipeline
import feature_engineering
import model_training
import prepare_grid


st.set_page_config(page_title="F1 Grand Prix Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ AI Grand Prix Simulator")
st.markdown("""
This engine automatically calculates driver momentum from historical data, requiring only 
Qualifying results to run a full 20-car race simulation via a C++ backend.
""")

with st.sidebar:
    st.header("📈 Model Info")
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

        st.metric("Model Type", metrics.get("model_type", "unknown").replace("_", " ").title())
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{metrics.get('mae', 'N/A')} pos")
        col2.metric("R² Score", f"{metrics.get('r2', 'N/A')}")

        st.metric("RMSE", f"{metrics.get('rmse', 'N/A')} pos")
        st.metric("Training Rows", metrics.get("n_training_rows", "N/A"))
        st.metric("Features", metrics.get("n_features", "N/A"))

        importances = metrics.get("feature_importances", {})
        if importances:
            st.subheader("Feature Importances")
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in sorted(importances.items(), key=lambda x: -x[1])
            ])
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

        cv_rf = metrics.get("cv_rf_mae")
        cv_gb = metrics.get("cv_gb_mae")
        if cv_rf and cv_gb:
            st.subheader("CV Model Comparison")
            st.caption("5-fold cross-validation MAE")
            cv_df = pd.DataFrame([
                {"Model": "Random Forest", "CV MAE": cv_rf},
                {"Model": "Gradient Boosting", "CV MAE": cv_gb},
            ])
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run a simulation to generate model metrics.")

st.divider()

st.subheader("📊 Weekend Qualifying Results")
try:
    with open(QUALI_FILE, "r") as f:
        quali_data = json.load(f)

    df_quali = pd.DataFrame(quali_data)
    df_quali.columns = ["Driver", "Grid Position"]
    st.dataframe(df_quali, use_container_width=True, hide_index=True)

except FileNotFoundError:
    st.error(f"Could not find {QUALI_FILE}. Please ensure the file exists in your data folder.")

st.divider()

if st.button("Run Full Simulation", type="primary", use_container_width=True):
    with st.spinner("Updating Model & Executing Simulation..."):
        try:
            with st.status("Checking pipeline...", expanded=False) as status:
                st.write("Checking raw data...")
                data_pipeline.run()

                st.write("Checking feature engineering...")
                feature_engineering.run()

                st.write("Checking model training...")
                model_training.run()

                st.write("Preparing starting grid...")
                prepare_grid.prepare_race_day_grid()

                status.update(label="Pipeline ready!", state="complete")

            if not os.path.exists(STARTING_GRID_FILE):
                st.error("Error: starting_grid.json was not created. Check prepare_grid.py output.")
                st.stop()
            if not os.path.exists(MODEL_METADATA_FILE):
                st.error("Error: model_metadata.json not found. Please run model training first.")
                st.stop()
            if not os.path.exists(INFERENCE_EXE):
                st.error(
                    f"Error: Inference binary not found at '{INFERENCE_EXE}'.\n\n"
                    f"Please compile it first:\n"
                    f"```\ng++ -O3 src_cpp/inference.cpp -o models/inference"
                    f"{'.exe' if platform.system() == 'Windows' else ''}\n```"
                )
                st.stop()

            result = subprocess.run(
                [INFERENCE_EXE],
                capture_output=True, text=True, encoding="utf-8",
                check=True,
                cwd=PROJECT_ROOT
            )

            st.success("Simulation Complete!")

            st.code(result.stdout, language="text")

            with open(STARTING_GRID_FILE, "r") as f:
                auto_grid = pd.DataFrame(json.load(f))
                auto_grid.columns = ["Driver", "Grid Position", "Momentum Score",
                                     "Racecraft Rating", "Constructor Strength", "Consistency"]

            with st.expander("View Auto-Calculated Input Math"):
                st.markdown("These are the final variables Python sent to the C++ Engine.")
                st.dataframe(auto_grid, use_container_width=True, hide_index=True)

        except FileNotFoundError:
            st.error(f"Engine Error: Could not find '{INFERENCE_EXE}'. Ensure you compiled it inside the models/ folder.")
        except subprocess.CalledProcessError as e:
            error_msg = f"Execution Error. The underlying script failed:\n\n**stdout:**\n```\n{e.stdout}\n```\n**stderr:**\n```\n{e.stderr}\n```"
            st.error(error_msg)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")