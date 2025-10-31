import numpy as np
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler     
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def train_and_evaluate_classifier():
    """Loads the LSM features, trains a Logistic Regression classifier, and evaluates it."""
    
    class_names = ["yes", "no", "up", "down", "backward", "stop", "bird", "cat", "nine",
                "eight", "zero", "follow"]
    dataset_filename = "lsm_features_larger.npz"

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

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    print("✅ Scaling complete.")

    print("\nTraining the Logistic Regression classifier...") 
    
    clf = LogisticRegression(
        multi_class="multinomial", 
        random_state=42, 
        max_iter=1000  
    ) 
    
    clf.fit(X_train_scaled, y_train) 
    print("✅ Training complete.")

    print("\nEvaluating performance on the test set...")
    y_pred = clf.predict(X_test_scaled) 
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print("\n" + "="*50)
    print("     FINAL RESULTS (with Logistic Regression)")
    print("="*50)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report)
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate_classifier()