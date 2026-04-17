import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

INPUT_FILE = os.path.join(DATA_DIR, "f1_engineered_data.csv")
OUTPUT_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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

    # =========================================================================
    # TASK 1: Chronological Grouped Time-Series Split
    # =========================================================================
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

    # =========================================================================
    # TASK 2a: Academic Baseline — DummyRegressor
    # =========================================================================
    print("\n--- Academic Baseline (DummyRegressor) ---")
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_mae = mean_absolute_error(y_test, y_pred_dummy)
    dummy_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
    print(f"  Dummy (Mean) MAE:   {dummy_mae:.2f} positions")
    print(f"  Dummy (Mean) RMSE:  {dummy_rmse:.2f} positions")

    # =========================================================================
    # Model Evaluation — 5-Fold Cross-Validation
    # =========================================================================
    print("\nEvaluating models with 5-fold cross-validation...\n")

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

    # Show improvement over baseline
    improvement = ((dummy_mae - mae) / dummy_mae) * 100
    print(f"\n  >> Model beats Dummy baseline by {improvement:.1f}% (MAE)")

    # =========================================================================
    # TASK 2b: Residual Analysis
    # =========================================================================
    residuals = np.abs(y_test.values - y_pred)
    within_2 = np.sum(residuals <= 2.0)
    pct_within_2 = (within_2 / len(residuals)) * 100

    print(f"\n--- Residual Analysis ---")
    print(f"  Predictions within ±2.0 grid positions: {within_2}/{len(residuals)} ({pct_within_2:.1f}%)")
    print(f"  Mean Absolute Residual:  {residuals.mean():.2f}")
    print(f"  Median Absolute Residual: {np.median(residuals):.2f}")
    print(f"  Max Absolute Residual:   {residuals.max():.2f}")

    # =========================================================================
    # Feature Importances
    # =========================================================================
    importances = dict(zip(FEATURE_COLUMNS, best_model.feature_importances_))
    print(f"\n--- Feature Importances ---")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:25s} {imp:.4f}")

    # =========================================================================
    # TASK 3: Generate Analytical Plots
    # =========================================================================
    print(f"\nGenerating analytical plots in '{PLOTS_DIR}'...")

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # --- Plot 1: Feature Importance Bar Chart ---
    sorted_features = sorted(importances.items(), key=lambda x: x[1])
    feat_names = [f[0] for f in sorted_features]
    feat_values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feat_names))
    ax.barh(feat_names, feat_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Importance Score", fontsize=13)
    ax.set_title("Feature Importance — Random Forest Regressor", fontsize=15, fontweight='bold')

    for i, v in enumerate(feat_values):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("  [OK] feature_importance.png saved")

    # --- Plot 2: Actual vs Predicted Scatter Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.35, edgecolors='black', linewidths=0.3,
               s=25, color='steelblue', label='Predictions')

    # Perfect prediction line (y=x)
    axis_min = min(y_test.min(), y_pred.min()) - 1
    axis_max = max(y_test.max(), y_pred.max()) + 1
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2,
            label='Perfect Prediction (y=x)')

    ax.set_xlabel("Actual Finish Position", fontsize=13)
    ax.set_ylabel("Predicted Finish Position", fontsize=13)
    ax.set_title("Actual vs Predicted — Test Set", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)
    print("  [OK] actual_vs_predicted.png saved")

    # --- Plot 3: Decision Tree Visualization ---
    if model_type == "random_forest":
        fig, ax = plt.subplots(figsize=(24, 10))
        plot_tree(
            best_model.estimators_[0],
            max_depth=3,
            feature_names=FEATURE_COLUMNS,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
            impurity=False
        )
        ax.set_title("Decision Tree Visualization (Tree #1, max_depth=3)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "tree_visualizer.png"), dpi=150)
        plt.close(fig)
        print("  [OK] tree_visualizer.png saved")
    else:
        # For GradientBoosting, estimators_ is an array of arrays
        fig, ax = plt.subplots(figsize=(24, 10))
        plot_tree(
            best_model.estimators_[0][0],
            max_depth=3,
            feature_names=FEATURE_COLUMNS,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
            impurity=False
        )
        ax.set_title("Decision Tree Visualization (Tree #1, max_depth=3)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "tree_visualizer.png"), dpi=150)
        plt.close(fig)
        print("  [OK] tree_visualizer.png saved")

    # =========================================================================
    # Export Model to JSON (for C++ inference engine)
    # =========================================================================
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

    # =========================================================================
    # Export Metrics Summary
    # =========================================================================
    metrics = {
        "model_type": model_type,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "n_training_rows": len(X_train),
        "n_testing_rows": len(X_test),
        "n_features": len(FEATURE_COLUMNS),
        "cv_rf_mae": round(rf_cv_mae, 2),
        "cv_gb_mae": round(gb_cv_mae, 2),
        "dummy_baseline_mae": round(dummy_mae, 2),
        "dummy_baseline_rmse": round(dummy_rmse, 2),
        "residual_within_2_pct": round(pct_within_2, 1),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel training complete.")
    print(f"  Model metadata: {output_json}")
    print(f"  Model metrics:  {METRICS_FILE}")
    print(f"  Plots:          {PLOTS_DIR}/")


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