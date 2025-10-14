import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def train_and_evaluate_classifier():
    """Loads the LSM features, trains a classifier, and evaluates it."""
    
    class_names = ["yes", "no", "up", "down"]
    dataset_filename = "lsm_features_dataset.npz"

    print(f"Loading feature dataset from '{dataset_filename}'...")
    if not Path(dataset_filename).exists():
        print(f"Error: Dataset file not found. Please run 'extract_lsm_features.py' first.")
        return
        
    data = np.load(dataset_filename)
    X_train = data['X_train_features']
    y_train = data['y_train']
    X_test = data['X_test_features']
    y_test = data['y_test']
    
    print(f"✅ Loaded {len(X_train)} training samples and {len(X_test)} test samples.")

    print("\nTraining the Random Forest classifier...") 
    
    clf = RandomForestClassifier(n_estimators=300, random_state=42) 
    
    clf.fit(X_train, y_train)
    print("✅ Training complete.")

    print("\nEvaluating performance on the test set...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print("\n" + "="*50)
    print("       FINAL RESULTS (with Random Forest)")
    print("="*50)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report)
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate_classifier()