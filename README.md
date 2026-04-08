# Project-Slipstream

## Quick Start

Run the commands from the repository root in this order:

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate the raw dataset:
   ```bash
   python src/data_pipeline.py
   ```
3. Create the engineered feature dataset:
   ```bash
   python src/feature_engineering.py
   ```
4. Train or retrain the model and update `models/model_metadata.json`:
   ```bash
   python src/model_training.py
   ```
5. Compile the C++ inference engine (from the repository root):
   ```bash
   g++ -O3 src_cpp/inference.cpp -o models/inference.exe
   ```
6. Run the Streamlit app:
   ```bash
   python -m streamlit run src/app.py
   ```

These steps ensure the data, model, and C++ inference engine are all generated before launching the app.
