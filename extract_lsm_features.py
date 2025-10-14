import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

NUM_NEURONS = 3000
NUM_OUTPUT_NEURONS = 500
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 7
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = 200

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

def extract_features(lsm, spike_data, desc=""):
    """Processes spike trains through the LSM and extracts MEAN SPIKE TIME features."""
    features = []
    
    all_feature_means = []
    all_feature_maxes = []

    for sample in tqdm(spike_data, desc=desc):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        
        current_features = feature_dict["mean_spike_times"]
        
        current_features[current_features < 0] = 0
        
        features.append(current_features)
        
        if np.any(current_features > 0): 
            all_feature_means.append(np.mean(current_features[current_features > 0]))
        all_feature_maxes.append(np.max(current_features))
            
    print(f"\n    --- DEBUG: Output Feature Statistics ({desc}) ---")
    if all_feature_means:
         print(f"    > Avg of Mean Spike Times (for active neurons): {np.mean(all_feature_means):.2f}")
    print(f"    > Overall Max Spike Time seen in any sample: {np.max(all_feature_maxes):.2f}")
    print(f"    --------------------------------------------------")

    return np.array(features)

def main():
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None: return
    I = np.mean(X_spikes)
    beta = SMALL_WORLD_K / 2
    w_critico = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) / beta
    print(f"\nCalculated Critical Weight (w_critico): {w_critico:.6f}\n")
    weights_mean = w_critico
    X_train, X_test, y_train, y_test = train_test_split(X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    print(f"Split data into {len(X_train)} training and {len(X_test)} test samples.")
    print("Instantiating SNN with critical weight...")
    params = SimulationParams(num_neurons=NUM_NEURONS, mean_weight=weights_mean, weight_variance=weights_mean * 5,
                              num_output_neurons=NUM_OUTPUT_NEURONS, is_random_uniform=False, membrane_threshold=MEMBRANE_THRESHOLD,
                              leak_coefficient=LEAK_COEFFICIENT, refractory_period=REFRACTORY_PERIOD,
                              small_world_graph_p=SMALL_WORLD_P, small_world_graph_k=SMALL_WORLD_K, input_spike_times=X_train[0])
    lsm = SNN(simulation_params=params)
    X_train_features = extract_features(lsm, X_train, desc="Extracting training features")
    X_test_features = extract_features(lsm, X_test, desc="Extracting test features")
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