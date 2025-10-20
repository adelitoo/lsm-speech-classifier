import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Constants from your LSM script ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 4
MEMBRANE_THRESHOLD = 2
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = 100
SPIKE_DATASET_FILE = "speech_spike_dataset.npz"

# --- Features to use for classification ---
FEATURES_TO_USE = ["mean_spike_times", "spike_counts"]

np.random.seed(42)

# --- Helper function from extract_lsm_features.py ---
def load_spike_dataset(filename=SPIKE_DATASET_FILE):
    """Loads the pre-processed spike train dataset."""
    print(f"Loading spike train dataset from '{filename}'...")
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        print("Please run your 'create_dataset.py' script first.")
        return None, None
    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples.")
    return X_spikes, y_labels

# --- Helper function from extract_lsm_features.py ---
def extract_features(lsm, spike_data, features_to_extract, desc=""):
    """
    Processes spike trains through the LSM and extracts a specified set of features.
    """
    all_combined_features = []
    
    # Optional: To track LSM activity, can be commented out for speed
    all_sample_active_neurons = [] 

    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        # Optional: Track activity
        spike_counts_for_sample = feature_dict["spike_counts"]
        all_sample_active_neurons.append(np.count_nonzero(spike_counts_for_sample))
        
        sample_feature_parts = []
        for feature_name in features_to_extract:
            feature_vector = feature_dict[feature_name].copy()
            
            # Clean up NaNs/Infs that might result from no spikes
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            
            sample_feature_parts.append(feature_vector)
        
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)
            
    # Optional: Print activity summary
    avg_active_neurons = np.mean(all_sample_active_neurons)
    print(f"    > Avg. active output neurons: {avg_active_neurons:.2f} / {lsm.num_output_neurons} ({avg_active_neurons/lsm.num_output_neurons*100:.2f}%)")

    return np.array(all_combined_features)

def run_single_experiment(X_train, y_train, X_test, y_test, current_weight):
    """
    Runs one full cycle of feature extraction and classification
    for a single given weight.
    """
    
    # 1. Instantiate SNN (LSM) with the new weight
    # We use a large variance as in your original script
    weight_variance = current_weight * 5 
    
    params = SimulationParams(num_neurons=NUM_NEURONS, mean_weight=current_weight, weight_variance=weight_variance,
                              num_output_neurons=NUM_OUTPUT_NEURONS, is_random_uniform=False, membrane_threshold=MEMBRANE_THRESHOLD,
                              leak_coefficient=LEAK_COEFFICIENT, refractory_period=REFRACTORY_PERIOD,
                              small_world_graph_p=SMALL_WORLD_P, small_world_graph_k=SMALL_WORLD_K, input_spike_times=X_train[0])
    lsm = SNN(simulation_params=params)

    # 2. Extract features for this specific LSM configuration
    X_train_features = extract_features(lsm, X_train, features_to_extract=FEATURES_TO_USE, desc="  Extracting train features")
    X_test_features = extract_features(lsm, X_test, features_to_extract=FEATURES_TO_USE, desc="  Extracting test features")

    # 3. Train and evaluate the classifier
    clf = RandomForestClassifier(n_estimators=350, random_state=42) 
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def main_experiment():
    """
    Main function to run the experiment loop.
    """
    # 1. Load the spike data once
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return

    # 2. Calculate the 'w_critico' from your script to use as a reference
    num_samples, num_input_neurons, num_time_bins = X_spikes.shape
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = SMALL_WORLD_K / 2
    w_critico = (MEMBRANE_THRESHOLD - 2 * I * REFRACTORY_PERIOD) / beta
    print(f"\nCalculated Reference 'w_critico': {w_critico:.6f}\n")

    # 3. Define the range of weights to test
    # We'll test a range *around* your w_critico
    # For example, from 25% to 250% of w_critico
    WEIGHTS_TO_TEST = np.linspace(w_critico * 0.1, w_critico * 4, num=500)
    
    print("Will test the following mean_weight values:")
    print([float(f"{w:.6f}") for w in WEIGHTS_TO_TEST])
    
    results = [] # To store (weight, accuracy) tuples

    # 4. Split data (we use the same split for all experiments)
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"\nUsing {len(X_train)} training and {len(X_test)} test samples for all experiments.")

    # 5. Run the main experiment loop
    print("\n" + "="*50)
    print("STARTING WEIGHT EXPERIMENT")
    print("="*50)
    
    for weight in tqdm(WEIGHTS_TO_TEST, desc="Overall Experiment Progress"):
        print(f"\n--- Testing weight: {weight:.6f} ---")
        
        accuracy = run_single_experiment(X_train, y_train, X_test, y_test, weight)
        
        print(f"  ✅ Final Accuracy for weight {weight:.6f}: {accuracy * 100:.2f}%")
        results.append((weight, accuracy))

    # 6. Process and plot results
    print("\n\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    
    # Unzip results for plotting
    weights = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    print("\nSummary of Results:")
    for w, acc in results:
        print(f"  Weight: {w:.6f}  ->  Accuracy: {acc * 100:.2f}%")

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.plot(weights, accuracies, 'o-', label="Test Accuracy", markersize=8)
    
    # Add a line for the reference w_critico
    plt.axvline(x=w_critico, color='r', linestyle='--', 
                label=f'Reference w_critico ({w_critico:.6f})')
                
    # Add a marker for the best accuracy
    best_idx = np.argmax(accuracies)
    best_w = weights[best_idx]
    best_acc = accuracies[best_idx]
    plt.plot(best_w, best_acc, 'r*', markersize=15, 
             label=f'Best Accuracy: {best_acc * 100:.2f}% at {best_w:.6f}')

    plt.title("LSM Classifier Accuracy vs. Average Synaptic Weight", fontsize=16)
    plt.xlabel("Average Synaptic Weight (mean_weight)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the plot
    plot_filename = "weight_vs_accuracy.png"
    plt.savefig(plot_filename)
    print(f"\n✅ Results plot saved to '{plot_filename}'")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main_experiment()