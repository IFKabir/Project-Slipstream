# Project Slipstream 🏎️

**AI-powered F1 Grand Prix race prediction engine** — uses a Python ML pipeline + C++ inference backend to simulate race outcomes from qualifying data.

## Features

- **Smart Data Pipeline** — Fetches F1 season data from FastF1, with automatic caching (no redundant re-fetches)
- **Feature Engineering** — Calculates rolling 3-race driver form averages
- **Random Forest Model** — Trains a scikit-learn model and exports decision trees as JSON
- **C++ Inference Engine** — Blazing-fast race simulation from exported model weights
- **Streamlit Dashboard** — One-click simulation with live pipeline status

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the C++ inference engine

**Linux / macOS:**
```bash
g++ -O3 src_cpp/inference.cpp -o models/inference
```

**Windows (MinGW / MSYS2):**
```bash
g++ -O3 src_cpp/inference.cpp -o models/inference.exe
```

### 3. Run the Streamlit app

```bash
python -m streamlit run src/app.py
```

That's it! The app automatically handles all data fetching, feature engineering, and model training on first run. Subsequent runs skip these steps unless the data changes.

### Manual Pipeline (Optional)

If you want to run individual pipeline steps:

```bash
python src/data_pipeline.py           # Fetch raw F1 data (skips if data exists)
python src/data_pipeline.py --force   # Force re-fetch
python src/feature_engineering.py     # Generate features (skips if up to date)
python src/model_training.py          # Train model (skips if up to date)
```

## Project Structure

```
Project-Slipstream/
├── src/
│   ├── app.py                  # Streamlit UI
│   ├── data_pipeline.py        # FastF1 data fetching
│   ├── feature_engineering.py  # Rolling averages, feature creation
│   ├── model_training.py       # Random Forest training + JSON export
│   └── prepare_grid.py         # Race-day grid preparation
├── src_cpp/
│   ├── inference.cpp           # C++ inference engine
│   └── include/nlohmann/       # JSON parsing library (header-only)
├── data/
│   ├── quali_results.json      # Input: qualifying positions
│   ├── starting_grid.json      # Auto-generated: grid + form scores
│   ├── f1_raw_data_master.csv  # Auto-generated: raw season data
│   └── f1_engineered_data.csv  # Auto-generated: features
├── models/
│   ├── model_metadata.json     # Auto-generated: exported RF trees
│   └── inference[.exe]         # Compiled binary (platform-specific)
├── tests/
│   ├── test_fastf1.py          # FastF1 smoke test (2024)
│   └── test_fastf1_2023.py     # FastF1 smoke test (2023)
└── requirements.txt
```

## Cross-Platform Notes

- All Python scripts use absolute paths resolved from script location — they work regardless of working directory
- The C++ binary resolves data/model paths relative to its own executable location
- On **Windows**: compile to `inference.exe`, on **Linux/macOS**: compile to `inference` (no extension)
- The Streamlit app auto-detects the correct binary name for the current platform
