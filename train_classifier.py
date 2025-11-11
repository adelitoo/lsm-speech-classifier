import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path


def train_and_evaluate_classifier():
    class_names = [
        "yes",
        "no",
        "up",
        "visual",
        "backward",
        "stop",
        "bird",
        "cat",
        "nine",
        "eight",
        "zero",
        "follow"]
    dataset_filename = "lsm_features_larger.npz"

    if not Path(dataset_filename).exists():
        print(f"Error: Dataset file not found. Please run 'extract_lsm_features.py' first.")
        return

    data = np.load(dataset_filename)
    X_train = data['X_train_features']
    y_train = data['y_train']
    X_test = data['X_test_features']
    y_test = data['y_test']

    print(f"Loaded {len(X_train)} training and {len(X_test)} test samples.")

    print("Training the Logistic Regression classifier...")
    clf = LogisticRegression(
        multi_class="multinomial",
        random_state=42,
        max_iter=1000
    )
    clf.fit(X_train, y_train)
    print("Training complete.")

    print("Evaluating performance on the test set...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print("\n--- Final Results ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    train_and_evaluate_classifier()
