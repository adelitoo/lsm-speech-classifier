"""
Test with just 6 very different classes to verify if the pipeline works at all.

If accuracy is still ~16% (random for 6 classes), the encoding is broken.
If accuracy is 50-70%, the encoding works but 35 classes is too hard.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Select 6 very acoustically different commands
SELECTED_COMMANDS = {
    "yes": 0,
    "no": 1,
    "left": 28,      # Changed from "up": very different from yes/no
    "stop": 32,      # Changed from "down": very different plosive
    "go": 14,        # Short, different from stop
    "nine": 20       # Different vowel sound
}

def load_original_spike_dataset():
    """Load the original spike dataset"""
    filename = "speech_spike_dataset_jittered.npz"
    if not Path(filename).exists():
        print(f"Error: {filename} not found")
        return None, None
    
    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    
    print(f"Loaded full dataset: {X_spikes.shape}")
    return X_spikes, y_labels

def filter_to_subset(X_spikes, y_labels):
    """Filter dataset to only include selected classes"""
    selected_indices = list(SELECTED_COMMANDS.values())
    mask = np.isin(y_labels, selected_indices)
    
    X_subset = X_spikes[mask]
    y_subset = y_labels[mask]
    
    # Remap labels to 0-5
    label_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_indices)}
    y_remapped = np.array([label_mapping[y] for y in y_subset])
    
    print(f"\nFiltered to {len(X_subset)} samples from {len(SELECTED_COMMANDS)} classes")
    print("Class distribution:")
    for name, orig_idx in SELECTED_COMMANDS.items():
        new_idx = label_mapping[orig_idx]
        count = np.sum(y_remapped == new_idx)
        print(f"  {name:10s} (class {new_idx}): {count} samples")
    
    return X_subset, y_remapped

def extract_features_for_subset(X_subset, y_subset):
    """Extract LSM features for the subset"""
    from snnpy.snn import SNN, SimulationParams
    from tqdm import tqdm
    
    # Use same parameters as main script
    NUM_NEURONS = 1000
    NUM_OUTPUT_NEURONS = 400
    LEAK_COEFFICIENT = 1 / 10000
    REFRACTORY_PERIOD = 2
    MEMBRANE_THRESHOLD = 2.0
    SMALL_WORLD_P = 0.2
    SMALL_WORLD_K = 200
    WEIGHT = 0.0056  # The optimal weight found
    
    print("\nCreating LSM...")
    params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=WEIGHT,
        weight_variance=WEIGHT * 0.1,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_subset[0]
    )
    
    lsm = SNN(simulation_params=params)
    
    print("Extracting features...")
    all_features = []
    
    for sample in tqdm(X_subset, desc="Processing"):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        # Extract same features as main script
        parts = []
        for key in ['spike_counts', 'mean_spike_times', 'first_spike_times', 
                    'last_spike_times', 'isi_mean', 'isi_variance', 
                    'spike_entropy', 'firing_rates', 'autocorrelation_first_lag',
                    'burstiness', 'spike_symmetry', 'burst_counts']:
            if key in feature_dict:
                vec = feature_dict[key].copy()
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                vec[vec < 0] = 0
                parts.append(vec)
        
        all_features.append(np.concatenate(parts))
    
    return np.array(all_features)

def plot_pca_subset(X_features, y_labels, output_file="pca_6_classes.png"):
    """Plot PCA for 6 classes - should show clearer separation if encoding works"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    # Create reverse mapping for labels
    idx_to_name = {idx: name for name, orig_idx in SELECTED_COMMANDS.items() 
                   for idx, orig in enumerate(SELECTED_COMMANDS.values()) if orig == orig_idx}
    
    plt.figure(figsize=(12, 9))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (name, _) in enumerate(SELECTED_COMMANDS.items()):
        mask = y_labels == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=colors[i], label=name, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    
    plt.legend(fontsize=12)
    plt.title(f"PCA of 6-Class Subset\nPC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%", 
              fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n✅ Saved PCA plot to '{output_file}'")
    plt.close()
    
    return scaler

def plot_confusion_matrix(y_true, y_pred, output_file="confusion_6_classes.png"):
    """Plot confusion matrix for 6 classes"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    class_names = list(SELECTED_COMMANDS.keys())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 6 Classes', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved confusion matrix to '{output_file}'")
    plt.close()

def main():
    print("="*60)
    print("TESTING WITH 6 ACOUSTICALLY DISTINCT CLASSES")
    print("="*60)
    print("\nThis test will reveal if your encoding preserves class information.")
    print("Random guessing: 16.7% accuracy")
    print("Working system: 50-70% accuracy")
    print("Current 35-class accuracy: 33%")
    print("\n" + "="*60 + "\n")
    
    # Load and filter data
    X_spikes, y_labels = load_original_spike_dataset()
    if X_spikes is None:
        return
    
    X_subset, y_subset = filter_to_subset(X_spikes, y_labels)
    
    # Extract features
    X_features = extract_features_for_subset(X_subset, y_subset)
    print(f"\nExtracted features shape: {X_features.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_subset, test_size=0.2, random_state=42, stratify=y_subset
    )
    
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot PCA
    plot_pca_subset(X_train_scaled, y_train)
    
    # Train classifiers
    print("\n" + "="*60)
    print("TRAINING CLASSIFIERS")
    print("="*60)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=10, random_state=42)
    }
    
    results = {}
    best_acc = 0
    best_name = None
    best_pred = None
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"  Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pred = y_pred
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nBest classifier: {best_name}")
    print(f"Accuracy: {best_acc*100:.2f}%\n")
    
    # Interpret results
    if best_acc < 0.25:
        print("❌ CRITICAL FAILURE (<25%)")
        print("   Your audio→spike encoding is fundamentally broken.")
        print("   The LSM cannot extract ANY meaningful class information.")
        print("   → You need to completely redesign your spike encoding.")
    elif best_acc < 0.40:
        print("⚠️  POOR PERFORMANCE (25-40%)")
        print("   Encoding captures some information but very weakly.")
        print("   → Consider: simpler rate coding, fewer mel bins, different thresholds")
    elif best_acc < 0.55:
        print("📊 MODERATE PERFORMANCE (40-55%)")
        print("   Encoding works but isn't optimal.")
        print("   → Fine-tune mel spectrogram parameters and spike thresholds")
    else:
        print("✅ GOOD PERFORMANCE (>55%)")
        print("   Encoding works! The 35-class problem is just hard.")
        print("   → Try: more samples per class, ensemble methods, feature selection")
    
    print("\nClassification Report:")
    print(classification_report(y_test, best_pred, 
                               target_names=list(SELECTED_COMMANDS.keys())))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, best_pred)
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()