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
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

QUALI_FILE = os.path.join(DATA_DIR, "quali_results.json")
STARTING_GRID_FILE = os.path.join(DATA_DIR, "starting_grid.json")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")
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

# ---- Actual 2024 Abu Dhabi GP Results (Dec 8, 2024) ----
ACTUAL_ABU_DHABI_2024 = [
    {"driver": "NOR", "actual_pos": 1},
    {"driver": "SAI", "actual_pos": 2},
    {"driver": "LEC", "actual_pos": 3},
    {"driver": "HAM", "actual_pos": 4},
    {"driver": "RUS", "actual_pos": 5},
    {"driver": "VER", "actual_pos": 6},
    {"driver": "GAS", "actual_pos": 7},
    {"driver": "HUL", "actual_pos": 8},
    {"driver": "ALO", "actual_pos": 9},
    {"driver": "PIA", "actual_pos": 10},
    {"driver": "ALB", "actual_pos": 11},
    {"driver": "TSU", "actual_pos": 12},
    {"driver": "ZHO", "actual_pos": 13},
    {"driver": "STR", "actual_pos": 14},
    {"driver": "DOO", "actual_pos": 15},
    {"driver": "MAG", "actual_pos": 16},
    {"driver": "LAW", "actual_pos": 17},
    {"driver": "BOT", "actual_pos": 18},
    {"driver": "PER", "actual_pos": 19},
    {"driver": "COL", "actual_pos": 20},
]

# All 20 F1 2024 drivers for the qualifying editor
ALL_DRIVERS_2024 = [
    "NOR", "PIA", "VER", "PER", "LEC", "SAI", "HAM", "RUS",
    "ALO", "STR", "GAS", "DOO", "HUL", "MAG", "TSU", "LAW",
    "ALB", "COL", "BOT", "ZHO"
]


st.set_page_config(page_title="F1 Grand Prix Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ AI Grand Prix Simulator")
st.markdown("""
This engine automatically calculates driver momentum from historical data, requiring only
Qualifying results to run a full 20-car race simulation via a C++ backend.
""")

# =====================================================================
# SIDEBAR — Model Info
# =====================================================================
with st.sidebar:
    st.header("📈 Model Info")
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

        st.metric("Model Type", metrics.get("model_type", "unknown").replace("_", " ").title())
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{metrics.get('mae', 'N/A')} pos")
        col2.metric("R2 Score", f"{metrics.get('r2', 'N/A')}")

        st.metric("RMSE", f"{metrics.get('rmse', 'N/A')} pos")
        st.metric("Training Rows", metrics.get("n_training_rows", "N/A"))
        st.metric("Features", metrics.get("n_features", "N/A"))

        best_params = metrics.get("best_params")
        if best_params:
            st.subheader("GridSearchCV Best Params")
            for k, v in best_params.items():
                st.text(f"  {k}: {v}")

        importances = metrics.get("feature_importances", {})
        if importances:
            st.subheader("Feature Importances")
            imp_df = pd.DataFrame([
                {"Feature": k, "Importance": v}
                for k, v in sorted(importances.items(), key=lambda x: -x[1])
            ])
            st.dataframe(imp_df, width="stretch", hide_index=True)

        cv_rf = metrics.get("cv_rf_mae")
        cv_gb = metrics.get("cv_gb_mae")
        if cv_rf and cv_gb:
            st.subheader("CV Model Comparison")
            st.caption("5-fold cross-validation MAE")
            cv_df = pd.DataFrame([
                {"Model": "Random Forest", "CV MAE": cv_rf},
                {"Model": "Gradient Boosting", "CV MAE": cv_gb},
            ])
            st.dataframe(cv_df, width="stretch", hide_index=True)

        dummy_mae = metrics.get("dummy_baseline_mae")
        if dummy_mae:
            st.subheader("Baseline Comparison")
            st.metric("Dummy MAE", f"{dummy_mae} pos")
            st.metric("Residual within +/-2", f"{metrics.get('residual_within_2_pct', 'N/A')}%")
    else:
        st.info("Run a simulation to generate model metrics.")

st.divider()

# =====================================================================
# TABS — Main Content
# =====================================================================
tab_sim, tab_quali, tab_compare, tab_plots = st.tabs([
    "🏁 Simulation", "📝 Edit Qualifying", "📊 Predicted vs Actual", "🖼️ Analytical Plots"
])

# =====================================================================
# TAB 1 — Simulation
# =====================================================================
with tab_sim:
    st.subheader("📊 Current Qualifying Grid")
    try:
        with open(QUALI_FILE, "r") as f:
            quali_data = json.load(f)
        df_quali = pd.DataFrame(quali_data)
        df_quali.columns = ["Driver", "Grid Position"]
        st.dataframe(df_quali, width="stretch", hide_index=True)
    except FileNotFoundError:
        st.warning("No qualifying data found. Use the 'Edit Qualifying' tab to enter grid positions.")

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

                # Show auto-calculated features
                with open(STARTING_GRID_FILE, "r") as f:
                    auto_grid = pd.DataFrame(json.load(f))

                with st.expander("View Auto-Calculated Input Features"):
                    st.markdown("These are the final feature values Python sent to the C++ Engine.")
                    st.dataframe(auto_grid, width="stretch", hide_index=True)

            except FileNotFoundError:
                st.error(f"Engine Error: Could not find '{INFERENCE_EXE}'. Ensure you compiled it inside the models/ folder.")
            except subprocess.CalledProcessError as e:
                error_msg = f"Execution Error. The underlying script failed:\n\n**stdout:**\n```\n{e.stdout}\n```\n**stderr:**\n```\n{e.stderr}\n```"
                st.error(error_msg)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# =====================================================================
# TAB 2 — Edit Qualifying Data
# =====================================================================
with tab_quali:
    st.subheader("📝 Edit Qualifying Grid Positions")
    st.markdown("""
    Enter the qualifying results below. Changes are saved to `quali_results.json` so the
    simulation uses your updated grid. Select each driver and their grid position.
    """)

    # Load existing data as defaults
    existing_quali = {}
    if os.path.exists(QUALI_FILE):
        try:
            with open(QUALI_FILE, "r") as f:
                for entry in json.load(f):
                    existing_quali[entry["driver"]] = entry["grid_pos"]
        except Exception:
            pass

    num_drivers = st.number_input("Number of drivers", min_value=1, max_value=20, value=20, step=1)

    quali_entries = []
    cols = st.columns(2)
    for i in range(num_drivers):
        col = cols[i % 2]
        with col:
            with st.container():
                c1, c2 = st.columns([2, 1])
                # Try to set sensible defaults from existing data
                sorted_existing = sorted(existing_quali.items(), key=lambda x: x[1])
                if i < len(sorted_existing):
                    default_driver = sorted_existing[i][0]
                    default_pos = sorted_existing[i][1]
                else:
                    default_driver = ALL_DRIVERS_2024[i] if i < len(ALL_DRIVERS_2024) else ALL_DRIVERS_2024[0]
                    default_pos = i + 1

                driver_idx = ALL_DRIVERS_2024.index(default_driver) if default_driver in ALL_DRIVERS_2024 else 0

                driver = c1.selectbox(
                    f"P{i+1} Driver",
                    ALL_DRIVERS_2024,
                    index=driver_idx,
                    key=f"driver_{i}"
                )
                grid_pos = c2.number_input(
                    f"Grid",
                    min_value=1,
                    max_value=20,
                    value=int(default_pos),
                    key=f"grid_{i}"
                )
                quali_entries.append({"driver": driver, "grid_pos": grid_pos})

    if st.button("Save Qualifying Data", type="primary", use_container_width=True):
        with open(QUALI_FILE, "w") as f:
            json.dump(quali_entries, f, indent=2)
        st.success(f"Qualifying data saved! {len(quali_entries)} drivers written to quali_results.json")
        st.rerun()

# =====================================================================
# TAB 3 — Predicted vs Actual Comparison
# =====================================================================
with tab_compare:
    st.subheader("📊 2024 Abu Dhabi GP: Predicted vs Actual Results")
    st.markdown("""
    Side-by-side comparison of the model's predictions against the **actual race results**
    from the 2024 Abu Dhabi Grand Prix (December 8, 2024).
    """)

    # Load predictions
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)

        df_pred = pd.DataFrame(predictions)
        df_actual = pd.DataFrame(ACTUAL_ABU_DHABI_2024)

        # Merge on driver
        df_compare = df_pred.merge(df_actual, on="driver", how="left")
        df_compare["delta"] = df_compare["position"] - df_compare["actual_pos"]
        df_compare["abs_delta"] = df_compare["delta"].abs()

        # Display table
        display_df = df_compare[["driver", "grid_pos", "position", "actual_pos", "delta"]].copy()
        display_df.columns = ["Driver", "Grid", "Predicted Pos", "Actual Pos", "Error"]
        display_df = display_df.sort_values("Predicted Pos")

        st.dataframe(
            display_df.style.map(
                lambda v: "color: green" if isinstance(v, (int, float)) and v == 0
                else ("color: orange" if isinstance(v, (int, float)) and abs(v) <= 2
                      else ""),
                subset=["Error"]
            ),
            width="stretch",
            hide_index=True,
            height=740
        )

        # Summary stats
        st.divider()
        col1, col2, col3, col4 = st.columns(4)

        avg_error = df_compare["abs_delta"].mean()
        exact_matches = (df_compare["delta"] == 0).sum()
        within_2 = (df_compare["abs_delta"] <= 2).sum()
        within_3 = (df_compare["abs_delta"] <= 3).sum()

        col1.metric("Avg Position Error", f"{avg_error:.1f}")
        col2.metric("Exact Matches", f"{exact_matches}/20")
        col3.metric("Within +/-2 Pos", f"{within_2}/20")
        col4.metric("Within +/-3 Pos", f"{within_3}/20")

        # Highlight notable predictions
        st.divider()
        st.subheader("Notable Predictions")

        best = df_compare.nsmallest(5, "abs_delta")[["driver", "position", "actual_pos", "delta"]]
        worst = df_compare.nlargest(5, "abs_delta")[["driver", "position", "actual_pos", "delta"]]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Best Predictions (closest to actual)**")
            best.columns = ["Driver", "Predicted", "Actual", "Error"]
            st.dataframe(best, width="stretch", hide_index=True)
        with c2:
            st.markdown("**Worst Predictions (furthest from actual)**")
            worst.columns = ["Driver", "Predicted", "Actual", "Error"]
            st.dataframe(worst, width="stretch", hide_index=True)

    else:
        st.info("No predictions found. Run a simulation first from the Simulation tab.")

# =====================================================================
# TAB 4 — Analytical Plots
# =====================================================================
with tab_plots:
    st.subheader("🖼️ Model Analytical Plots")
    st.markdown("Generated during model training. These visualize key model characteristics.")

    plot_files = {
        "Feature Importance": "feature_importance.png",
        "Actual vs Predicted": "actual_vs_predicted.png",
        "Decision Tree Visualization": "tree_visualizer.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Learning Curve": "learning_curve.png",
    }

    available_plots = {name: path for name, path in plot_files.items()
                       if os.path.exists(os.path.join(PLOTS_DIR, path))}

    if available_plots:
        # Show plots in a 2-column layout
        plot_names = list(available_plots.keys())
        for i in range(0, len(plot_names), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(plot_names):
                    name = plot_names[idx]
                    path = os.path.join(PLOTS_DIR, available_plots[name])
                    with col:
                        st.markdown(f"**{name}**")
                        st.image(path, use_container_width=True)

        # If odd number, last plot already shown in the loop
    else:
        st.info("No plots found. Run a simulation to generate analytical plots.")