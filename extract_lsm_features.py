import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# --- Constants (no changes here) ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 5
MEMBRANE_THRESHOLD = 2
SMALL_WORLD_P = 0.1
SMALL_WORLD_K = 100

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

# --- MODIFICATIONS START HERE ---

def extract_features(lsm, spike_data, features_to_extract, desc=""):
    """
    Processes spike trains through the LSM and extracts a specified set of features.
    """
    all_combined_features = []
    
    # Lists to store activity statistics for each sample
    all_sample_total_spikes = []
    all_sample_active_neurons = []

    for sample in tqdm(spike_data, desc=desc):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        # --- Collect debugging stats (no changes to this part) ---
        spike_counts_for_sample = feature_dict["spike_counts"]
        total_spikes_in_sample = np.sum(spike_counts_for_sample)
        active_neurons_in_sample = np.count_nonzero(spike_counts_for_sample)
        all_sample_total_spikes.append(total_spikes_in_sample)
        all_sample_active_neurons.append(active_neurons_in_sample)
        
        # --- NEW: Extract and combine the specified features ---
        sample_feature_parts = []
        for feature_name in features_to_extract:
            # Get the feature vector from the dictionary
            feature_vector = feature_dict[feature_name].copy()
            
            # Clean up potential invalid values (NaNs, inf, negatives)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            
            sample_feature_parts.append(feature_vector)
        
        # Concatenate all parts (e.g., spike_counts + mean_spike_times) into one vector
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)
            
    # Print the aggregated debugging statistics (no changes to this part)
    num_output_neurons = lsm.num_output_neurons
    avg_total_spikes = np.mean(all_sample_total_spikes)
    avg_active_neurons = np.mean(all_sample_active_neurons)
    
    print(f"\n    --- DEBUG: LSM Activity Statistics ({desc}) ---")
    print(f"    > Avg. total spikes per sample: {avg_total_spikes:.2f}")
    print(f"    > Avg. active output neurons per sample: {avg_active_neurons:.2f} / {num_output_neurons} ({avg_active_neurons/num_output_neurons*100:.2f}%)")
    print(f"    > Min/Max active neurons in a sample: {np.min(all_sample_active_neurons)} / {np.max(all_sample_active_neurons)}")
    print(f"    --------------------------------------------------")

    return np.array(all_combined_features)

def main():
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None: return
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = SMALL_WORLD_K / 2
    w_critico = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) / beta
    print(f"\nCalculated Critical Weight (w_critico): {w_critico:.6f}\n")
    
    weights_mean = w_critico 
    print(f"Using mean weight: {weights_mean:.6f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    print(f"Split data into {len(X_train)} training and {len(X_test)} test samples.")
    print("Instantiating SNN...")
    params = SimulationParams(num_neurons=NUM_NEURONS, mean_weight=weights_mean, weight_variance=weights_mean * 5,
                              num_output_neurons=NUM_OUTPUT_NEURONS, is_random_uniform=False, membrane_threshold=MEMBRANE_THRESHOLD,
                              leak_coefficient=LEAK_COEFFICIENT, refractory_period=REFRACTORY_PERIOD,
                              small_world_graph_p=SMALL_WORLD_P, small_world_graph_k=SMALL_WORLD_K, input_spike_times=X_train[0])
    lsm = SNN(simulation_params=params)
    
    # --- NEW: Define which features to use for this experiment ---
    features_to_use = ["mean_spike_times", "spike_counts"]
    print(f"\nExtracting features: {features_to_use}")
    
    # Pass the list of features to the extraction function
    X_train_features = extract_features(lsm, X_train, features_to_extract=features_to_use, desc="Extracting training features")
    X_test_features = extract_features(lsm, X_test, features_to_extract=features_to_use, desc="Extracting test features")
    
    print("\nFeature extraction complete.")
    print(f"Shape of training features: {X_train_features.shape}")
    print(f"Shape of test features: {X_test_features.shape}")
    
    output_filename = "lsm_features_dataset.npz"
    print(f"\nSaving final features and labels to '{output_filename}'...")
    np.savez_compressed(output_filename, X_train_features=X_train_features, y_train=y_train,
                        X_test_features=X_test_features, y_test=y_test)
    print("\n✅ Process complete. Your feature dataset is ready for classification.")

if __name__ == "__main__":
    main()