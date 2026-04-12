import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- ABSOLUTE PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

INPUT_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")
OUTPUT_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)

# All 5 features used for prediction
FEATURE_COLUMNS = ['GridPosition', 'Momentum_Score', 'Racecraft_Rating',
                    'Constructor_Strength', 'Consistency']


def train_f1_model(input_csv, output_json):
    df = pd.read_csv(input_csv)

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in data. Filling with 0.")
            df[col] = 0.0

    X = df[FEATURE_COLUMNS]
    y = df['FinalPosition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Selection via Cross-Validation ---
    print("Evaluating models with 5-fold cross-validation...\n")

    # Candidate 1: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5,
                                    scoring='neg_mean_absolute_error')
    rf_cv_mae = -rf_cv_scores.mean()
    print(f"  Random Forest    — CV MAE: {rf_cv_mae:.2f} (±{rf_cv_scores.std():.2f})")

    # Candidate 2: Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5,
                                    scoring='neg_mean_absolute_error')
    gb_cv_mae = -gb_cv_scores.mean()
    print(f"  Gradient Boosting — CV MAE: {gb_cv_mae:.2f} (±{gb_cv_scores.std():.2f})")

    # Pick the best model
    if rf_cv_mae <= gb_cv_mae:
        print(f"\n[OK] Selected: Random Forest (lower CV MAE)")
        best_model = rf_model
        model_type = "random_forest"
    else:
        print(f"\n[OK] Selected: Gradient Boosting (lower CV MAE)")
        best_model = gb_model
        model_type = "gradient_boosting"

    # --- Final Training on Full Train Set ---
    print(f"\nTraining {model_type} on full training set...")
    best_model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  Mean Absolute Error:  {mae:.2f} positions")
    print(f"  Root Mean Sq Error:   {rmse:.2f} positions")
    print(f"  R² Score:             {r2:.3f}")

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, best_model.feature_importances_))
    print(f"\n--- Feature Importances ---")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.4f}")

    # --- Export Model as JSON for C++ Inference ---
    # For both RF and GB, we export the tree ensemble in the same format
    export_data = {
        "model_type": model_type,
        "n_estimators": len(best_model.estimators_) if model_type == "random_forest" else best_model.n_estimators,
        "feature_names": list(FEATURE_COLUMNS),
        "trees": []
    }

    def extract_tree(tree):
        """Extract a single decision tree into a JSON-serializable format."""
        tree_ = tree.tree_
        nodes = []
        for i in range(tree_.node_count):
            is_leaf = tree_.children_left[i] == tree_.children_right[i]
            node = {
                "feature": int(tree_.feature[i]) if not is_leaf else -1,
                "threshold": float(tree_.threshold[i]) if not is_leaf else 0.0,
                "left": int(tree_.children_left[i]) if not is_leaf else -1,
                "right": int(tree_.children_right[i]) if not is_leaf else -1,
                "is_leaf": bool(is_leaf),
                "prob": float(tree_.value[i][0][0])
            }
            nodes.append(node)
        return {"nodes": nodes}

    if model_type == "random_forest":
        for estimator in best_model.estimators_:
            export_data["trees"].append(extract_tree(estimator))
    else:
        # Gradient Boosting stores trees as an array of arrays
        # Each estimator is a 1D array of single-output trees
        export_data["learning_rate"] = best_model.learning_rate
        export_data["init_value"] = float(best_model.init_.constant_[0])
        for estimator_arr in best_model.estimators_:
            export_data["trees"].append(extract_tree(estimator_arr[0]))

    with open(output_json, "w") as f:
        json.dump(export_data, f)

    # --- Save Metrics ---
    metrics = {
        "model_type": model_type,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "n_training_rows": len(X_train),
        "n_features": len(FEATURE_COLUMNS),
        "cv_rf_mae": round(rf_cv_mae, 2),
        "cv_gb_mae": round(gb_cv_mae, 2),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel training complete.")
    print(f"  Model metadata: {output_json}")
    print(f"  Model metrics:  {METRICS_FILE}")


def run(force=False):
    """Run model training. Skips if model is up to date unless force=True."""
    if os.path.exists(OUTPUT_FILE) and not force:
        if os.path.exists(INPUT_FILE):
            input_mtime = os.path.getmtime(INPUT_FILE)
            output_mtime = os.path.getmtime(OUTPUT_FILE)
            if output_mtime >= input_mtime:
                print(f"Model is up to date. Skipping. Use --force to re-train.")
                return
    train_f1_model(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    force = "--force" in sys.argv
    run(force=force)