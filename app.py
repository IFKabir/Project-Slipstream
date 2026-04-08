import streamlit as st
import subprocess

# --- UI Configuration ---
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Podium Predictor AI")
st.markdown("""
Welcome to the predictive analytics engine. This tool uses a machine learning model trained on historical data, 
with a high-performance **C++ inference backend**, to calculate the probability of a driver finishing in the Top 3.
""")

st.divider()

# --- Input Section ---
st.subheader("Configure Driver Profile")

col1, col2 = st.columns(2)

with col1:
    grid_pos = st.number_input("Starting Grid Position", min_value=1, max_value=20, value=3, step=1)
    
    # 1 for Yes, 0 for No
    top_car_input = st.radio("Is the driver in a Top-Tier Car? (e.g., Red Bull, Ferrari, McLaren)", ["Yes", "No"])
    top_car = 1 if top_car_input == "Yes" else 0

with col2:
    recent_form = st.number_input("Recent Form (Avg finish in last 3 races)", min_value=1.0, max_value=20.0, value=4.5, step=0.1)

st.divider()

# --- Execution Section ---
if st.button("Calculate Probability", type="primary", use_container_width=True):
    
    with st.spinner("Initializing C++ Inference Engine..."):
        try:
            # 1. The Bridge: Python calls the C++ executable with the user's inputs
            # Note: On Windows it's 'inference.exe'. On Mac/Linux it would be './f1_engine'
            command = ["inference.exe", str(grid_pos), str(recent_form), str(top_car)]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            # 2. Parse the output from C++
            raw_output = result.stdout.strip()
            
            # Attempt to convert to float
            probability_decimal = float(raw_output)
            probability_percent = probability_decimal * 100

            # 3. Display the Results Beautifully
            st.success("Inference complete.")
            
            st.subheader("Results")
            st.metric(label="Podium Probability", value=f"{probability_percent:.1f}%")
            
            # Visual progress bar 
            st.progress(probability_decimal)
            
            # Provide some dynamic text feedback
            if probability_percent > 50:
                st.info("🟢 High chance of a podium finish. Strong starting position and form.")
            elif probability_percent > 15:
                st.warning("🟡 Moderate chance. Will likely need strategy luck or overtakes to reach the podium.")
            else:
                st.error("🔴 Low chance. A podium is mathematically unlikely without chaotic race events.")

        except ValueError:
            st.error(f"Engine Error: C++ returned unexpected output. Raw output: {raw_output}")
        except FileNotFoundError:
            st.error("Engine Error: Could not find 'inference.exe'. Ensure it is compiled and in the same folder as app.py.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")