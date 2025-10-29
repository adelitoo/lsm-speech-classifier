"""
LSM Weight Sweep Experiment (run_weight_sweep.py)

This script loops through multiple synaptic weights to find the
optimal operating point (criticality) for accuracy.

It automatically calculates the theoretical 'w_critico' based on
the network parameters and input data, then tests in sub-critical,
critical, and super-critical regimes.

It now includes plotting at the end.
"""

import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt  # <-- ADDED IMPORT

# --- Network Parameters ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 4
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.2
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2) # k=200

# Define the feature set to use for the experiment
# 'original' is a good, fast choice. 'all' is slow but includes everything.
FEATURE_SET_TO_TEST = 'original'

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
    (Based on your supervisor's 'checks' function)
    """
    # Calculate I (average input rate per neuron per timestep)
    # We estimate 'I' over the first 500 training samples
    num_samples = min(500, len(input_data)) 
    
    total_spikes = 0
    total_elements = 0
    
    # Assuming input_data is (samples, neurons, time)
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

    # This is the formula from your supervisor's 'checks' function
    numerator = (lsm_params.membrane_threshold - 2 * avg_I * (lsm_params.refractory_period))
    w_critico = numerator / beta
    
    print("\n--- Theoretical Calculation ---")
    print(f"  Avg Input Rate (I): {avg_I:.6f} (spikes/neuron/timestep)")
    print(f"  Connectivity (beta): {beta:.1f} (k/2)")
    print(f"  Refractory Period: {lsm_params.refractory_period}")
    print(f"  Threshold: {lsm_params.membrane_threshold}")
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
        
        # Concatenate requested features
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
    print(f"    Mean Activity: {activity_pct:.1f}%")
    return np.array(all_features), activity_pct


# --- Classifier Training Function ---

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, quiet=False):
    """
    Trains and evaluates the classifier, returning the test accuracy.
    (This function is from your train_classifier.py)
    """
    if not quiet:
        print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    
    if not quiet:
        print("\nTraining the Logistic Regression classifier...") 
    
    clf = LogisticRegression(
        multi_class="multinomial", 
        random_state=42, 
        max_iter=1000,
        solver='lbfgs' # Using lbfgs as it's a good default
    ) 
    clf.fit(X_train_scaled, y_train) 
    
    if not quiet:
        print("✅ Training complete.")
        print("\nEvaluating performance on the test set...")
    
    y_pred = clf.predict(X_test_scaled) 
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the single most important metric for the sweep
    return accuracy


# --- Single Experiment Function ---

def run_experiment_for_weight(mean_weight, X_train, y_train, X_test, y_test, base_params):
    """
    Runs the full extraction and training pipeline for a *single* weight.
    """
    # Set the specific weight for this experiment
    base_params.mean_weight = mean_weight
    base_params.weight_variance = mean_weight * 0.1 # Keep variance proportional
    
    # We must create a new SNN object for each weight
    # to ensure the network is re-initialized correctly.
    lsm = SNN(simulation_params=base_params)
    
    feature_keys = FEATURE_SETS[FEATURE_SET_TO_TEST]
    n_features = len(feature_keys) * NUM_OUTPUT_NEURONS
    
    print(f"Extracting features ({FEATURE_SET_TO_TEST}, {n_features} dims)...")

    # Extract features for train and test sets
    X_train_feat, train_act = extract_all_features(
        lsm, X_train, feature_keys, "Training"
    )
    X_test_feat, test_act = extract_all_features(
        lsm, X_test, feature_keys, "Testing"
    )
    
    print(f"Training and evaluating for weight={mean_weight:.8f}...")
    
    # Train and get accuracy
    test_accuracy = train_and_evaluate_classifier(
        X_train_feat, y_train, X_test_feat, y_test,
        quiet=True # Don't print the full report every loop
    )
    
    # Return the key metrics
    avg_activity = (train_act + test_act) / 2
    return mean_weight, avg_activity, test_accuracy


# --- Main Experiment Runner ---

if __name__ == "__main__":
    
    # 1. Load data ONCE
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        exit()
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"\nData split: {len(y_train)} train, {len(y_test)} test samples.")
    
    
    # 2. Create a base SimulationParams object
    # This holds all the *fixed* parameters of our network.
    # We use a placeholder weight (0.0) since it's not needed for the calculation.
    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0, # Placeholder, will be set in the loop
        weight_variance=0.0, # Placeholder, will be set in the loop
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0] # Just for initialization shape
    )
    
    # 3. CALCULATE w_critico using the formula
    w_critico_calculated = calculate_theoretical_w_critico(base_params, X_train)

    
    # 4. Define multipliers to test *around* your w_critico
    # You updated this to a wider range
    multipliers_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                           1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # 5. Generate the final list of weights to test
    weights_to_test = [w_critico_calculated * m for m in multipliers_to_test]
    
    print(f"\nStarting weight sweep based on calculated w_critico = {w_critico_calculated:.8f}")
    print(f"Testing weights: {[float(f'{w:.8f}') for w in weights_to_test]}")
    print(f"Feature set: '{FEATURE_SET_TO_TEST}'")
    
    results = [] # To store (weight, activity, accuracy)
    
    # 6. Run the loop
    for i, weight in enumerate(weights_to_test):
        print("\n" + "="*60)
        print(f"TESTING WEIGHT: {weight:.8f} ({multipliers_to_test[i]:.1f} * w_critico)")
        print("="*60)
        
        # We pass the base_params to the function
        w, act, acc = run_experiment_for_weight(
            weight, X_train, y_train, X_test, y_test, base_params
        )
        
        results.append((w, act, acc))
        print(f"✅ Result: Weight={w:.8f}, Activity={act:.1f}%, Accuracy={acc*100:.2f}%")

    # 7. Print final summary
    print("\n\n" + "="*60)
    print("WEIGHT SWEEP FINAL RESULTS")
    print("="*60)
    print(f"Base w_critico (calculated): {w_critico_calculated:.8f}")
    print(f"Feature Set: '{FEATURE_SET_TO_TEST}'")
    print(f"{'Weight':<14} | {'Activity (%)':<12} | {'Accuracy (%)':<12}")
    print("-"*42)
    
    # Find the best result
    best_result = max(results, key=lambda item: item[2])
    
    for w, act, acc in results:
        marker = "⭐ BEST" if w == best_result[0] else ""
        print(f"{w:<14.8f} | {act:<12.1f} | {acc*100:<12.2f} {marker}")
        
    print("="*60)
    print(f"Optimal accuracy of {best_result[2]*100:.2f}% found at:")
    print(f"  Weight = {best_result[0]:.8f}")
    print(f"  Activity = {best_result[1]:.1f}%")
    print("="*60)


    # --- 8. PLOTTING SECTION ---
    
    print("\nGenerating plots...")

    # Unpack results for plotting
    plot_weights = np.array([r[0] for r in results])
    plot_activities = np.array([r[1] for r in results])
    plot_accuracies = np.array([r[2] for r in results])
    
    best_weight, best_activity, best_accuracy = best_result

    # --- PLOT 1: Accuracy vs. Synaptic Weight ---
    plt.figure(figsize=(10, 6))
    plt.plot(plot_weights, plot_accuracies * 100, marker='o', linestyle='-', label='Test Accuracy')
    
    # Highlight the best point
    plt.scatter(best_weight, best_accuracy * 100, color='red', s=100, zorder=5, label=f'Best: {best_accuracy*100:.2f}%')
    
    # Mark the theoretical w_critico
    plt.axvline(x=w_critico_calculated, color='k', linestyle='--', label=f'w_critico ({w_critico_calculated:.6f})')
    
    plt.title('LSM Performance: Accuracy vs. Synaptic Weight')
    plt.xlabel('Mean Synaptic Weight (w)')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lsm_accuracy_vs_weight.png')
    print("✅ Saved 'lsm_accuracy_vs_weight.png'")

    # --- PLOT 2: Activity vs. Synaptic Weight ---
    plt.figure(figsize=(10, 6))
    plt.plot(plot_weights, plot_activities, marker='s', linestyle='-', color='green', label='Network Activity')
    
    # Highlight the activity at the best accuracy point
    plt.scatter(best_weight, best_activity, color='red', s=100, zorder=5, label=f'Activity at Best Acc: {best_activity:.1f}%')
    
    # Mark the theoretical w_critico
    plt.axvline(x=w_critico_calculated, color='k', linestyle='--', label=f'w_critico ({w_critico_calculated:.6f})')
    
    plt.title('LSM Dynamics: Activity vs. Synaptic Weight')
    plt.xlabel('Mean Synaptic Weight (w)')
    plt.ylabel('Mean Output Activity (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lsm_activity_vs_weight.png')
    print("✅ Saved 'lsm_activity_vs_weight.png'")