"""
LSM Feature Extraction (Configurable Multiplier)

This script extracts features from a Liquid State Machine.
It calculates the theoretical w_critico and uses a 
COMMAND-LINE MULTIPLIER to set the final synaptic weight.

(This version has all plotting code removed for faster runs)
"""

import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA  <-- REMOVED
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt  <-- REMOVED
from tqdm import tqdm
from pathlib import Path
import argparse 

# --- Network Parameters ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 4
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2) # k=200

# --- Feature Set Definitions ---
FEATURE_SETS = {
    'all': ['spike_counts', 'spike_variances', 'mean_spike_times', 
            'first_spike_times', 'last_spike_times', 'mean_isi', 
            'isi_variances', 'burst_counts'],
    'rate': ['spike_counts', 'spike_variances', 'burst_counts'],
    'timing': ['mean_spike_times', 'first_spike_times', 'last_spike_times'],
    'rhythm': ['mean_isi', 'isi_variances'],
    'original': ['spike_counts', 'spike_variances', 'mean_spike_times', 
                 'mean_isi', 'isi_variances']
}

np.random.seed(42)

# --- Theoretical w_critico Function ---

def calculate_theoretical_w_critico(lsm_params, input_data):
    """
    Calculates the theoretical w_critico based on network params and input data.
    """
    # Estimate 'I' over the first 500 training samples
    num_samples = min(500, len(input_data)) 
    
    total_spikes = 0
    total_elements = 0
    
    for sample in input_data[:num_samples]:
        total_spikes += np.sum(sample)
        total_elements += (sample.shape[0] * sample.shape[1]) # Neurons * Time

    if total_elements == 0:
        print("⚠️ Error calculating 'I': No spike data found.")
        return 0.007 # Failsafe
        
    avg_I = total_spikes / total_elements
    beta = lsm_params.small_world_graph_k / 2
    
    if beta == 0:
        print("⚠️ Error calculating 'beta': small_world_graph_k is zero.")
        return 0.007 # Failsafe

    numerator = (lsm_params.membrane_threshold - 2 * avg_I * (lsm_params.refractory_period))
    w_critico = numerator / beta
    
    print("\n--- Theoretical Calculation ---")
    print(f"  Avg Input Rate (I): {avg_I:.6f} (spikes/neuron/timestep)")
    print(f"  Connectivity (beta): {beta:.1f} (k/2)")
    print(f"  Calculated w_critico: {w_critico:.8f}")
    print("-------------------------------")
    
    return w_critico

# --- Data Loading Function ---

def load_spike_dataset(filename="speech_spike_dataset_pure_redundancy.npz"):
    """Load spike train dataset"""
    print(f"Loading '{filename}'...")
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
        print("❌ Error: No dataset found")
        return None, None
    
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples, shape {X_spikes.shape}")
    print(f"   Avg spikes per sample: {np.sum(X_spikes)/len(X_spikes):.1f}")
    return X_spikes, y_labels

# --- Feature Extraction Function ---

def extract_all_features(lsm, spike_data, feature_keys, desc=""):
    """Extracts features for a given dataset and LSM"""
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
        for key in feature_keys:
            if key in feature_dict:
                vec = np.nan_to_num(
                    feature_dict[key].copy(), 
                    nan=0.0, posinf=0.0, neginf=0.0
                )
                vec[vec < 0] = 0
                parts.append(vec)
        
        all_features.append(np.concatenate(parts))
        
    activity_pct = np.mean(all_active) / lsm.num_output_neurons * 100
    print(f"\n    Activity ({desc}): {activity_pct:.1f}%")
    print(f"    Active neurons: {np.mean(all_active):.1f}/{lsm.num_output_neurons}")
    return np.array(all_features), activity_pct

# --- Visualization Function (REMOVED) ---
# def check_separability(...):
#     ...

# --- Main Execution ---

def main(feature_set: str, multiplier: float):
    
    # 1. Load data
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    print(f"\nDataset: {X_spikes.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # 2. Create base params object to calculate w_critico
    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0, # Placeholder
        weight_variance=0.0, # Placeholder
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0] # For shape info
    )
            
    # 3. Calculate theoretical w_critico
    w_critico_calculated = calculate_theoretical_w_critico(base_params, X_train)
            
    # 4. Set the optimal weight based on the COMMAND-LINE multiplier
    optimal_weight = w_critico_calculated * multiplier
    optimal_mult = multiplier
            
    print(f"\nUsing weight multiplier: {optimal_mult:.2f}")
    print(f"  w_critico (theoretical): {w_critico_calculated:.8f}")
    print(f"  FINAL WEIGHT USED: {optimal_weight:.8f}")

    # 5. Create the final LSM with the optimal weight
    print(f"\nCreating LSM ({NUM_NEURONS} neurons, {NUM_OUTPUT_NEURONS} outputs)...")
    
    base_params.mean_weight = optimal_weight
    base_params.weight_variance = optimal_weight * 0.1 # Standard variance
            
    lsm = SNN(simulation_params=base_params) # Create the final LSM
    
    # 6. Extract features
    feature_keys = FEATURE_SETS[feature_set]
    print(f"\nExtracting feature set: '{feature_set}' ({len(feature_keys)} features)")
    print(f"   Features: {feature_keys}")

    X_train_feat, train_act = extract_all_features(lsm, X_train, feature_keys, "Training")
    X_test_feat, test_act = extract_all_features(lsm, X_test, feature_keys, "Testing")
    
    print(f"\nFeatures: train={X_train_feat.shape}, test={X_test_feat.shape}")
    
    # --- 7. Scale Features (MOVED FROM check_separability) ---
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    print("✅ Scaling complete.")
    
    # 8. Save features
    # (Using the dynamic filename from our previous version)
    output_file = f"lsm_features_{feature_set}_mult_{optimal_mult:.2f}.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_features=X_train_scaled,
        y_train=y_train,
        X_test_features=X_test_scaled,
        y_test=y_test,
        weight_multiplier=optimal_mult,
        final_weight=optimal_weight,
        w_critico=w_critico_calculated,
        train_activity_pct=train_act,
        test_activity_pct=test_act,
        feature_set=feature_set,
        feature_keys=feature_keys
    )
    
    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Final Weight: {optimal_mult:.2f} * w_critico = {optimal_weight:.8f}")
    print(f"Activity: train={train_act:.1f}%, test={test_act:.1f}%")
    print(f"Feature dims: {X_train_feat.shape[1]}")
    print(f"Saved to: {output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from a spike train dataset using an LSM."
    )
    parser.add_argument(
        "--feature-set", 
        type=str, 
        default="original",
        choices=FEATURE_SETS.keys(),
        help="The set of features to extract (default: original)."
    )
    parser.add_argument(
        "--multiplier", 
        type=float, 
        default=0.6, # Default to the best one you found
        help="Multiplier for w_critico (e.g., 0.6 for 60%%)"
    )
    
    args = parser.parse_args()
    
    main(feature_set=args.feature_set, multiplier=args.multiplier)