import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# TUNED PARAMETERS - Finding the balance
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400      # Reduced from 400
LEAK_COEFFICIENT = 0        # Non-zero for decay
REFRACTORY_PERIOD = 4
MEMBRANE_THRESHOLD = 2
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = 100            # Reduced connectivity

# Key tuning parameter - we'll try multiple values
WEIGHT_MULTIPLIERS_TO_TRY = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]  # Try range around critical

np.random.seed(42)

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    print(f"Loading spike train dataset from '{filename}'...")
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples.")
    return X_spikes, y_labels


def extract_features(lsm, spike_data, features_to_extract, desc=""):
    """
    Processes spike trains through the LSM and extracts a specified set of features.
    """
    all_combined_features = []
    
    all_sample_total_spikes = []
    all_sample_active_neurons = []

    for sample in tqdm(spike_data, desc=desc):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        spike_counts_for_sample = feature_dict["spike_counts"]
        total_spikes_in_sample = np.sum(spike_counts_for_sample)
        active_neurons_in_sample = np.count_nonzero(spike_counts_for_sample)
        all_sample_total_spikes.append(total_spikes_in_sample)
        all_sample_active_neurons.append(active_neurons_in_sample)
        
        sample_feature_parts = []
        for feature_name in features_to_extract:
            feature_vector = feature_dict[feature_name].copy()
            
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            
            sample_feature_parts.append(feature_vector)
        
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)
            
    num_output_neurons = lsm.num_output_neurons
    avg_total_spikes = np.mean(all_sample_total_spikes)
    avg_active_neurons = np.mean(all_sample_active_neurons)
    activity_pct = avg_active_neurons / num_output_neurons * 100
    
    print(f"\n    --- DEBUG: LSM Activity Statistics ({desc}) ---")
    print(f"    > Avg. total spikes per sample: {avg_total_spikes:.2f}")
    print(f"    > Avg. active output neurons per sample: {avg_active_neurons:.2f} / {num_output_neurons} ({activity_pct:.2f}%)")
    print(f"    > Min/Max active neurons in a sample: {np.min(all_sample_active_neurons)} / {np.max(all_sample_active_neurons)}")
    
    # WARNING if still saturated
    if activity_pct > 85:
        print(f"    ⚠️  WARNING: LSM is over-saturated (>85% active)!")
        print(f"    ⚠️  Try LOWER weight multiplier")
    elif activity_pct < 30:
        print(f"    ⚠️  WARNING: LSM is under-stimulated (<30% active)")
        print(f"    ⚠️  Try HIGHER weight multiplier")
    else:
        print(f"    ✅ LSM activity looks healthy (30-85% active)")
    
    print(f"    --------------------------------------------------")

    return np.array(all_combined_features), activity_pct


def check_feature_separability(X_features, y_labels, output_file="feature_separability.png"):
    """Quick diagnostic to check if features separate classes"""
    print("\n📊 Checking feature separability with PCA...")
    
    # Use subset for speed
    n_samples = min(2000, len(X_features))
    indices = np.random.choice(len(X_features), n_samples, replace=False)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_features[indices])
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_labels[indices], 
                         cmap='tab20', alpha=0.5, s=20, edgecolors='k', linewidth=0.3)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f"PCA of LSM Features ({n_samples} samples)\nGood separation = distinct clusters")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved separability plot to '{output_file}'")
    print(f"   PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"   PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
    plt.close()


def quick_test_weight(X_sample, y_sample, weights_mean, num_samples=500):
    """
    Quickly test a weight value on a small sample to check activity level.
    Returns average activity percentage.
    """
    params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=weights_mean,
        weight_variance=weights_mean * 0.5,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_sample[0]
    )
    lsm = SNN(simulation_params=params)
    
    active_neurons_list = []
    for sample in X_sample[:num_samples]:
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        spike_counts = feature_dict["spike_counts"]
        active_neurons = np.count_nonzero(spike_counts)
        active_neurons_list.append(active_neurons)
    
    avg_active = np.mean(active_neurons_list)
    activity_pct = avg_active / NUM_OUTPUT_NEURONS * 100
    return activity_pct


def find_optimal_weight(X_train, y_train, w_critico):
    """
    Test different weight multipliers to find one in the 30-85% activity range.
    """
    print("\n" + "="*60)
    print("🔍 SEARCHING FOR OPTIMAL WEIGHT")
    print("="*60)
    print("Testing different weight multipliers on a sample of data...")
    print(f"Target: 40-70% neuron activity\n")
    
    results = []
    for multiplier in WEIGHT_MULTIPLIERS_TO_TRY:
        test_weight = w_critico * multiplier
        print(f"Testing multiplier {multiplier:.2f} (weight={test_weight:.6f})... ", end="", flush=True)
        
        activity = quick_test_weight(X_train, y_train, test_weight, num_samples=300)
        results.append((multiplier, test_weight, activity))
        
        status = "✅ GOOD" if 30 <= activity <= 85 else ("⚠️ LOW" if activity < 30 else "⚠️ HIGH")
        print(f"{activity:.1f}% active {status}")
    
    # Find best multiplier (closest to 55% target)
    target = 55
    best_multiplier, best_weight, best_activity = min(results, key=lambda x: abs(x[2] - target))
    
    print("\n" + "="*60)
    print(f"📊 SELECTED: multiplier={best_multiplier:.2f}, weight={best_weight:.6f}")
    print(f"   Expected activity: {best_activity:.1f}%")
    print("="*60 + "\n")
    
    return best_weight, best_multiplier


def main():
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None: return
    
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    
    # Calculate critical weight
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = SMALL_WORLD_K / 2
    w_critico = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) / beta
    
    print("\n" + "="*60)
    print("INITIAL WEIGHT CALCULATION")
    print("="*60)
    print(f"Calculated Critical Weight (w_critico): {w_critico:.6f}")
    print(f"Input spikes per sample: {np.sum(X_spikes) / num_samples:.1f}")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"\nSplit data into {len(X_train)} training and {len(X_test)} test samples.")
    
    # Find optimal weight
    optimal_weight, optimal_multiplier = find_optimal_weight(X_train, y_train, w_critico)
    
    # Instantiate SNN with optimal weight
    print("Instantiating final SNN with optimal parameters...")
    params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=optimal_weight,
        weight_variance=optimal_weight * 0.5,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0]
    )
    lsm = SNN(simulation_params=params)
    
    # Extract features
    features_to_use = ["mean_spike_times", "spike_counts"]
    print(f"\nExtracting features: {features_to_use}")
    
    X_train_features, train_activity = extract_features(
        lsm, X_train, 
        features_to_extract=features_to_use, 
        desc="Extracting training features"
    )
    
    X_test_features, test_activity = extract_features(
        lsm, X_test, 
        features_to_extract=features_to_use, 
        desc="Extracting test features"
    )
    
    print("\nFeature extraction complete.")
    print(f"Shape of training features: {X_train_features.shape}")
    print(f"Shape of test features: {X_test_features.shape}")
    
    # Check feature quality
    check_feature_separability(X_train_features, y_train)
    
    # Save results with metadata
    output_filename = "lsm_features_dataset.npz"
    print(f"\nSaving final features and labels to '{output_filename}'...")
    np.savez_compressed(
        output_filename,
        X_train_features=X_train_features,
        y_train=y_train,
        X_test_features=X_test_features,
        y_test=y_test,
        # Save configuration for reference
        weight_multiplier=optimal_multiplier,
        final_weight=optimal_weight,
        train_activity_pct=train_activity,
        test_activity_pct=test_activity
    )
    print("\n✅ Process complete. Your feature dataset is ready for classification.")
    print(f"\nFinal configuration:")
    print(f"  - Weight multiplier: {optimal_multiplier:.2f}")
    print(f"  - Mean weight: {optimal_weight:.6f}")
    print(f"  - Train activity: {train_activity:.1f}%")
    print(f"  - Test activity: {test_activity:.1f}%")

if __name__ == "__main__":
    main()