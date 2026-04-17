# 🏎️ Project Slipstream — F1 Race Prediction Engine

> **AI Lab Final Submission** — An end-to-end machine learning pipeline that predicts Formula 1 Grand Prix race finishing positions using historical race data, ensemble regression models, and a compiled C++ inference backend.

---

## 📌 Project Overview

**Project Slipstream** is an AI-powered F1 race prediction engine that combines a Python-based ML training pipeline with a high-performance C++ inference engine. The system:

1. **Fetches** historical F1 race data (2010–2024) via the FastF1 API.
2. **Engineers** five predictive features: `GridPosition`, `Momentum_Score`, `Racecraft_Rating`, `Constructor_Strength`, and `Consistency`.
3. **Trains** a `RandomForestRegressor` and a `GradientBoostingRegressor`, selecting the best model via 5-Fold Cross-Validation.
4. **Evaluates** against a `DummyRegressor` baseline to academically prove the model's predictive power beyond random guessing.
5. **Exports** the trained decision trees as a JSON structure, consumed by a compiled C++ binary for real-time inference.

### Key Design Decision — Avoiding Look-Ahead Bias

Unlike a naïve random `train_test_split`, this project uses a **Chronological Grouped Time-Series Split**:

- The dataset is sorted by `Year` and `Round`.
- For **every season**, the first 80% of races form the training set and the final 20% form the test set.
- This prevents any future race data from leaking into the training set, ensuring the model is always evaluated on races it has never "seen" — mirroring real-world deployment.

---

## 🗂️ Project Structure

```
Project-Slipstream/
├── src/
│   ├── app.py                  # Streamlit Dashboard (one-click UI)
│   ├── data_pipeline.py        # FastF1 data fetching + caching
│   ├── feature_engineering.py  # Rolling averages, feature creation
│   ├── model_training.py       # ML training, evaluation, plots
│   ├── merge_data.py           # Data merging utilities
│   └── prepare_grid.py         # Race-day grid preparation (features for inference)
├── src_cpp/
│   ├── inference.cpp           # C++ inference engine (tree traversal)
│   └── include/nlohmann/       # Header-only JSON library
├── data/
│   ├── f1_raw_data_master.csv  # Raw season data (auto-generated)
│   ├── f1_engineered_data.csv  # Engineered features (auto-generated)
│   ├── quali_results.json      # Qualifying positions input
│   └── starting_grid.json      # Grid + features for C++ engine
├── models/
│   ├── model_metadata.json     # Exported decision trees (JSON)
│   ├── model_metrics.json      # Evaluation metrics summary
│   └── inference[.exe]         # Compiled C++ binary
├── plots/
│   ├── feature_importance.png  # Feature weight visualization
│   ├── actual_vs_predicted.png # Scatter: actual vs predicted
│   └── tree_visualizer.png     # Decision tree diagram (depth=3)
├── tests/
│   ├── test_fastf1.py          # FastF1 smoke test (2024)
│   └── test_fastf1_2023.py     # FastF1 smoke test (2023)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Execution Instructions

### Prerequisites

- **Python 3.10+** with `pip`
- **g++** (MinGW/MSYS2 on Windows, GCC on Linux/macOS) for C++ compilation

### Step 1 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run the ML Training Pipeline

This fetches data, engineers features, trains the model, generates plots, and exports the decision trees:

```bash
# Run the full data pipeline (fetches & caches F1 data)
python src/data_pipeline.py

# Generate engineered features
python src/feature_engineering.py

# Train the model (evaluates RF vs GB, generates plots/)
python src/model_training.py
```

To force re-execution of any step (skip the "up to date" cache check):

```bash
python src/data_pipeline.py --force
python src/feature_engineering.py --force
python src/model_training.py --force
```

### Step 3 — Prepare the Starting Grid for Inference

```bash
python src/prepare_grid.py
```

This reads `data/quali_results.json`, computes all 5 model features for each driver from their historical data, and outputs `data/starting_grid.json`.

### Step 4 — Compile & Run the C++ Inference Engine

**Windows (MinGW / MSYS2):**
```bash
g++ -O3 src_cpp/inference.cpp -o models/inference.exe
models\inference.exe
```

**Linux / macOS:**
```bash
g++ -O3 src_cpp/inference.cpp -o models/inference
./models/inference
```

The engine reads `models/model_metadata.json` and `data/starting_grid.json`, performs tree traversal across all estimators, and prints the predicted race classification to stdout. Results are also saved to `data/predictions.json`.

### Step 5 (Optional) — Launch the Streamlit Dashboard

```bash
python -m streamlit run src/app.py
```

The dashboard handles the entire pipeline automatically on first launch.

---

## 🧪 Model Evaluation Summary

The training script (`model_training.py`) outputs the following analysis:

| Metric | Description |
|--------|-------------|
| **Dummy Baseline MAE/RMSE** | Performance of a `DummyRegressor(strategy='mean')` — proves the model is not just guessing. |
| **Cross-Validation MAE** | 5-Fold CV scores for both Random Forest and Gradient Boosting. |
| **Test Set MAE / RMSE / R²** | Final evaluation on the held-out chronological test set. |
| **Residual Analysis** | Percentage of predictions within ±2.0 grid positions of actual results. |
| **Feature Importances** | Relative contribution of each of the 5 features. |

### Generated Plots (saved to `plots/`)

| Plot | File | Purpose |
|------|------|---------|
| Feature Importance | `feature_importance.png` | Horizontal bar chart showing the weight of each feature. |
| Actual vs Predicted | `actual_vs_predicted.png` | Scatter plot with a red dashed y=x line showing prediction variance. |
| Tree Visualizer | `tree_visualizer.png` | Graphical export of the first decision tree (depth-limited to 3). |

---

## 🔧 Technical Details

### Features Used (5 Total)

| Feature | Description |
|---------|-------------|
| `GridPosition` | Qualifying/grid position for the race |
| `Momentum_Score` | Exponentially weighted moving average of recent points scored |
| `Racecraft_Rating` | Average positions gained/lost vs grid position (recent 3 races) |
| `Constructor_Strength` | Team performance metric based on aggregate team results |
| `Consistency` | Standard deviation of recent finishing positions (lower = more consistent) |

### Models Compared

| Model | Configuration |
|-------|--------------|
| `RandomForestRegressor` | 150 estimators, max_depth=8, min_samples_split=5, min_samples_leaf=3 |
| `GradientBoostingRegressor` | 200 estimators, max_depth=5, lr=0.1, subsample=0.8 |
| `DummyRegressor` | Baseline (strategy='mean') |

### C++ Inference Engine

The C++ binary (`inference.cpp`) reconstructs the ensemble model from the exported JSON tree structure. For Random Forests, it averages predictions across all trees. For Gradient Boosting, it applies the learned learning rate and initial value for additive prediction. The engine uses the header-only [nlohmann/json](https://github.com/nlohmann/json) library for JSON parsing.

---

## 🌐 Cross-Platform Notes

- All Python scripts resolve paths relative to the script location — they work regardless of CWD.
- The C++ binary resolves data/model paths relative to its own executable location.
- On **Windows**: compile to `inference.exe`. On **Linux/macOS**: compile to `inference` (no extension).
- The Streamlit app auto-detects the correct binary name for the platform.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
