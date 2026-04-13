import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

INPUT_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")
OUTPUT_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLUMNS = ['GridPosition', 'Momentum_Score', 'Racecraft_Rating',
                    'Constructor_Strength', 'Consistency']


def train_f1_model(input_csv, output_json):
    df = pd.read_csv(input_csv)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in data. Filling with 0.")
            df[col] = 0.0

    # Ensure 'Round' column exists for chronological sorting
    if 'Round' not in df.columns:
        import fastf1
        CACHE_DIR = os.path.join(PROJECT_ROOT, "f1_cache")
        fastf1.Cache.enable_cache(CACHE_DIR)
        df['Round'] = 0
        for y in df['Year'].unique():
            try:
                schedule = fastf1.get_event_schedule(int(y))
                idx = df['Year'] == y
                df.loc[idx, 'Round'] = df.loc[idx, 'RaceName'].map(
                    dict(zip(schedule['EventName'], schedule['RoundNumber']))
                ).fillna(0).astype(int)
            except Exception as e:
                print(f"Failed to fetch schedule for {y}: {e}")

    # 1. Ensure data is perfectly chronological first
    df = df.sort_values(['Year', 'Round']) 
    
    train_dfs = []
    test_dfs = []

    # 2. Loop through the dataset, isolating one year at a time
    for year, group in df.groupby('Year'):
        # Find out how many total races happened this year (e.g., 19 in 2010, 24 in 2024)
        total_races = group['Round'].nunique()
        
        # Calculate where the 80% cutoff is for this specific calendar
        cutoff_round = int(total_races * 0.8)
        
        # Split the year into the first 80% (Train) and final 20% (Test)
        train_year = group[group['Round'] <= cutoff_round]
        test_year = group[group['Round'] > cutoff_round]
        
        train_dfs.append(train_year)
        test_dfs.append(test_year)

    # 3. Glue all the training years together, and all the testing years together
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    # 4. Separate features (X) and targets (y)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['FinalPosition']
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['FinalPosition']
    
    print(f"\nCustom 80/20 Yearly Split Complete:")
    print(f"  Training rows: {len(X_train)} (First 80% of every season)")
    print(f"  Testing rows:  {len(X_test)} (Final 20% of every season)")

    print("Evaluating models with 5-fold cross-validation...\n")

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

    if rf_cv_mae <= gb_cv_mae:
        print(f"\n[OK] Selected: Random Forest (lower CV MAE)")
        best_model = rf_model
        model_type = "random_forest"
    else:
        print(f"\n[OK] Selected: Gradient Boosting (lower CV MAE)")
        best_model = gb_model
        model_type = "gradient_boosting"

    print(f"\nTraining {model_type} on full training set...")
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Metrics ---")
    print(f"  Mean Absolute Error:  {mae:.2f} positions")
    print(f"  Root Mean Sq Error:   {rmse:.2f} positions")
    print(f"  R² Score:             {r2:.3f}")

    importances = dict(zip(FEATURE_COLUMNS, best_model.feature_importances_))
    print(f"\n--- Feature Importances ---")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.4f}")

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
        export_data["learning_rate"] = best_model.learning_rate
        export_data["init_value"] = float(best_model.init_.constant_[0])
        for estimator_arr in best_model.estimators_:
            export_data["trees"].append(extract_tree(estimator_arr[0]))

    with open(output_json, "w") as f:
        json.dump(export_data, f)

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