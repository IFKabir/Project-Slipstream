import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_f1_model(input_csv):
    df = pd.read_csv(input_csv)
    
    X = df[['GridPosition', 'Recent_3_Race_Avg']]
    y = df['FinalPosition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Regressor instead of Classifier to predict continuous finish position
    print("Training the Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\nMean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} grid positions")
    
    # --- EXPORT MODEL LOGIC TO JSON FOR C++ ---
    export_data = {
        "n_estimators": len(model.estimators_),
        "feature_names": list(X.columns),
        "trees": []
    }

    def extract_tree(tree):
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

    for estimator in model.estimators_:
        export_data["trees"].append(extract_tree(estimator))

    with open("model_metadata.json", "w") as f:
        json.dump(export_data, f)
    
    print("\nModel training complete. Metadata saved to model_metadata.json")

if __name__ == "__main__":
    train_f1_model("f1_engineered_data.csv")