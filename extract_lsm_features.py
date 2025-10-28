import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# LARGER NETWORK for more complex representations
NUM_NEURONS = 1000  # Increased from 1000
NUM_OUTPUT_NEURONS = 400  # Increased from 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 4
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)  

BASE_MEAN_WEIGHT = 0.007
WEIGHT_MULTIPLIERS = [1.0]

np.random.seed(42)

def load_spike_dataset(filename="speech_spike_dataset_pure_redundancy.npz"):
    print(f"Loading '{filename}'...")
    
    # Try different dataset files in order of preference
    filenames_to_try = [
        "speech_spike_dataset_pure_redundancy.npz",
        "speech_spike_dataset_jittered.npz",
        "speech_spike_dataset_rate_coded.npz"
    ]
    
    data = None
    for fname in filenames_to_try:
        if Path(fname).exists():
            print(f"✅ Found '{fname}'")
            data = np.load(fname)
            filename = fname
            break
    
    if data is None:
        print("Error: No dataset found")
        return None, None
    
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples, shape {X_spikes.shape}")
    print(f"   Avg spikes: {np.sum(X_spikes)/len(X_spikes):.1f}")
    return X_spikes, y_labels

def extract_all_features(lsm, spike_data, desc=""):
    all_features = []
    all_active = []
    
    for sample in tqdm(spike_data, desc=desc):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        spike_counts = feature_dict["spike_counts"]
        all_active.append(np.count_nonzero(spike_counts))
        
        parts = []
        for key in ['spike_counts', 'mean_spike_times', 'first_spike_times', 
                    'last_spike_times', 'isi_mean', 'isi_variance', 
                    'spike_entropy', 'firing_rates', 'autocorrelation_first_lag',
                    'burstiness', 'spike_symmetry', 'burst_counts']:
            if key in feature_dict:
                vec = np.nan_to_num(feature_dict[key].copy(), nan=0.0, posinf=0.0, neginf=0.0)
                vec[vec < 0] = 0
                parts.append(vec)
        
        all_features.append(np.concatenate(parts))
    
    activity_pct = np.mean(all_active) / lsm.num_output_neurons * 100
    print(f"\n    Activity ({desc}): {activity_pct:.1f}%")
    print(f"    Active neurons: {np.mean(all_active):.1f}/{lsm.num_output_neurons}")
    
    return np.array(all_features), activity_pct

def quick_test_weight(X_sample, weight, num_samples=200):
    params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=weight,
        weight_variance=weight * 0.1,
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
    
    active_list = []
    for sample in X_sample[:num_samples]:
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        active_list.append(np.count_nonzero(lsm.extract_features_from_spikes()["spike_counts"]))
    
    return np.mean(active_list) / NUM_OUTPUT_NEURONS * 100

def find_optimal_weight(X_train):
    print("\n" + "="*60)
    print("SEARCHING FOR OPTIMAL WEIGHT (Larger Network)")
    print("="*60)
    
    results = []
    for mult in WEIGHT_MULTIPLIERS:
        weight = BASE_MEAN_WEIGHT * mult
        print(f"Testing {mult:.2f}x (w={weight:.6f})... ", end="", flush=True)
        activity = quick_test_weight(X_train, weight)
        results.append((mult, weight, activity))
        status = "✅" if 40 <= activity <= 75 else "⚠️"
        print(f"{activity:.1f}% {status}")
    
    best = min(results, key=lambda x: abs(x[2] - 55))
    print(f"\n📊 Selected: {best[0]:.2f}x, w={best[1]:.6f}, activity={best[2]:.1f}%")
    print("="*60 + "\n")
    return best[1], best[0]

def check_separability(X_features, y_labels, output_file="pca_larger_lsm.png"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    n_samples = min(3000, len(X_features))
    indices = np.random.choice(len(X_features), n_samples, replace=False)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled[indices])
    
    plt.figure(figsize=(14, 10))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_labels[indices],
               cmap='tab20', alpha=0.6, s=30, edgecolors='k', linewidth=0.4)
    plt.colorbar(label='Class')
    plt.title(f"PCA - Larger LSM ({NUM_NEURONS}N, {NUM_OUTPUT_NEURONS}O)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved to '{output_file}'")
    print(f"   PC1+PC2: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
    plt.close()
    return scaler

def main():
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    print(f"\nDataset: {X_spikes.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    optimal_weight, optimal_mult = find_optimal_weight(X_train)
    
    print(f"Creating larger LSM ({NUM_NEURONS} neurons, {NUM_OUTPUT_NEURONS} outputs)...")
    params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=optimal_weight,
        weight_variance=optimal_weight * 0.1,
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
    
    print("\nExtracting features...")
    X_train_feat, train_act = extract_all_features(lsm, X_train, "Training")
    X_test_feat, test_act = extract_all_features(lsm, X_test, "Testing")
    
    print(f"\nFeatures: train={X_train_feat.shape}, test={X_test_feat.shape}")
    
    scaler = check_separability(X_train_feat, y_train)
    
    X_train_scaled = scaler.transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    
    output_file = "lsm_features_larger.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_features=X_train_scaled,
        y_train=y_train,
        X_test_features=X_test_scaled,
        y_test=y_test,
        weight_multiplier=optimal_mult,
        final_weight=optimal_weight,
        train_activity_pct=train_act,
        test_activity_pct=test_act
    )
    
    print("\n✅ Complete!")
    print(f"Weight: {optimal_mult:.2f}x = {optimal_weight:.6f}")
    print(f"Activity: {train_act:.1f}% / {test_act:.1f}%")
    print(f"Feature dims: {X_train_feat.shape[1]}")

if __name__ == "__main__":
    main()