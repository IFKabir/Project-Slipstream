import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_f1_model(input_csv):
    df = pd.read_csv(input_csv)
    
    X = df[['GridPosition', 'Recent_3_Race_Avg', 'Front_Runner_Car']]
    y = df['IsPodium']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    print("Training the Random Forest model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Feature Importance (The 'Why') ---")
    importances = model.feature_importances_
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.4f}")
        
    export_data = {
        "n_estimators": len(model.estimators_),
        "feature_names": list(X.columns),
        "classes": model.classes_.tolist()
    }
    with open("model_metadata.json", "w") as f:
        json.dump(export_data, f)
    
    print("\nModel training complete. Metadata saved to model_metadata.json")
    return model

if __name__ == "__main__":
    train_f1_model("f1_engineered_data.csv")