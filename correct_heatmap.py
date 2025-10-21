import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Base configuration (used as a template) ---
BASE_CONFIG = {
    'num_neurons': 1000,
    'num_output_neurons': 400,
    'leak_coefficient': 0,
    'refractory_period': 4,
    'membrane_threshold': 2, # This will be overridden in the loop
    'small_world_p': 0.2,    # This will be overridden in the loop
    'small_world_k': 100
}

# Feature extraction options
USE_BOTH_FEATURES = True  # Using both mean_spike_times and spike_counts

# --- Helper Functions ---

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    return data['X_spikes'], data['y_labels']

def extract_features(lsm, spike_data, use_both_features=True, desc=""):
    features = []
    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        
        if use_both_features:
            mean_times = feature_dict["mean_spike_times"].copy()
            spike_counts = feature_dict["spike_counts"].copy()
            
            mean_times = np.nan_to_num(mean_times, nan=0.0, posinf=0.0, neginf=0.0)
            spike_counts = np.nan_to_num(spike_counts, nan=0.0, posinf=0.0, neginf=0.0)
            
            mean_times[mean_times < 0] = 0
            spike_counts[spike_counts < 0] = 0
            
            combined = np.concatenate([mean_times, spike_counts])
            features.append(combined)
        else:
            mean_times = feature_dict["mean_spike_times"].copy()
            mean_times = np.nan_to_num(mean_times, nan=0.0, posinf=0.0, neginf=0.0)
            mean_times[mean_times < 0] = 0
            features.append(mean_times)
    
    return np.array(features)

def calculate_I_and_I_eff(X_spikes, config):
    """Calculates I (avg input) and I_eff (effective input term)."""
    num_samples, num_input_neurons, num_time_bins = X_spikes.shape
    
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    
    # This is the "external electricity" term
    I_eff = 2 * I * config['refractory_period'] 
    
    return I, I_eff

def is_valid_configuration(config, I):
    """Check if a configuration will produce valid weights."""
    beta = config['small_world_k'] / 2
    if beta == 0: return False, "beta (k/2) is zero"
    
    w_critico = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
    
    if w_critico <= 0:
        return False, f"Critical weight would be {w_critico:.6f} (must be > 0)"
    
    return True, "Valid"

def test_configuration(config, X_train, X_test, y_train, y_test, I):
    """Test a single parameter configuration and return accuracy."""
    try:
        is_valid, message = is_valid_configuration(config, I)
        if not is_valid:
            # Return None instead of 0.0 so invalid configs are masked in heatmap
            return None, None

        beta = config['small_world_k'] / 2
        weights_mean = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
        
        params = SimulationParams(
            num_neurons=config['num_neurons'],
            mean_weight=weights_mean,
            weight_variance=weights_mean * 5,
            num_output_neurons=config['num_output_neurons'],
            is_random_uniform=False,
            membrane_threshold=config['membrane_threshold'],
            leak_coefficient=config['leak_coefficient'],
            refractory_period=config['refractory_period'],
            small_world_graph_p=config['small_world_p'],
            small_world_graph_k=config['small_world_k'],
            input_spike_times=X_train[0]
        )
        
        lsm = SNN(simulation_params=params)
        
        X_train_features = extract_features(lsm, X_train, USE_BOTH_FEATURES, desc="Train")
        X_test_features = extract_features(lsm, X_test, USE_BOTH_FEATURES, desc="Test")
        
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, y_train)
        
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, weights_mean
    
    except Exception as e:
        print(f"  ❌ Error with config {config}: {e}")
        return None, None

# --- Main Grid Search Function ---

def run_grid_search():
    """
    Runs a 2D grid search over 'small_world_p' (beta) and 
    'membrane_threshold' (theta).
    """
    print("="*70)
    print("STARTING 2D GRID SEARCH (small_world_p vs. membrane_threshold)")
    print("="*70)

    print("Loading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples\n")

    # 1. Calculate the "external electricity" (I_eff)
    I, I_eff = calculate_I_and_I_eff(X_spikes, BASE_CONFIG)
    print(f"Calculated average input I: {I:.6f}")
    print(f"Calculated 'External Current' (I_eff = 2*I*ref_period): {I_eff:.6f}\n")

    # 2. Define the parameter ranges based on supervisor's instructions
    
    # Beta from 0.1 to 0.4 with increment of 0.05
    p_range = np.arange(0.1, 0.4 + 0.05, 0.05)
    
    # Theta: "a bit below the external current and then raise it up to at least 10 times"
    # Start from 0.8*I_eff (a bit below) up to 10*I_eff
    threshold_range = np.linspace(I_eff * 0.8, I_eff * 10, num=15)
    
    print("Testing Parameters:")
    print(f"  'beta' (small_world_p): {[round(p, 2) for p in p_range]}")
    print(f"  'theta' (membrane_threshold): {[round(t, 3) for t in threshold_range]}")
    print(f"  Range: {threshold_range[0]:.3f} (0.8×I_eff) to {threshold_range[-1]:.3f} (10×I_eff)")
    print(f"\nTotal configurations to test: {len(p_range) * len(threshold_range)}\n")

    results = []
    invalid_count = 0
    
    # 3. Run the nested loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for p_val in tqdm(p_range, desc="Overall Progress (small_world_p)"):
            for t_val in threshold_range:
                
                config = BASE_CONFIG.copy()
                config['small_world_p'] = p_val
                config['membrane_threshold'] = t_val
                
                accuracy, w_crit = test_configuration(config, X_train, X_test, y_train, y_test, I)
                
                if accuracy is None:
                    invalid_count += 1
                
                results.append({
                    'small_world_p': p_val,
                    'membrane_threshold': t_val,
                    'accuracy': accuracy,
                    'critical_weight': w_crit
                })

    # 4. Process results and create the heatmap
    print(f"\nGrid search complete. {invalid_count} invalid configurations found.")
    print("Processing results...")
    
    if not results:
        print("Error: No results were generated. Check configurations.")
        return

    df = pd.DataFrame(results)
    
    # Show some statistics
    valid_results = df[df['accuracy'].notna()]
    if len(valid_results) > 0:
        print(f"\n✅ Valid configurations: {len(valid_results)}/{len(df)}")
        print(f"   Accuracy range: {valid_results['accuracy'].min():.4f} - {valid_results['accuracy'].max():.4f}")
        print(f"   Mean accuracy: {valid_results['accuracy'].mean():.4f}")
        best_config = valid_results.loc[valid_results['accuracy'].idxmax()]
        print(f"   Best config: beta={best_config['small_world_p']:.2f}, theta={best_config['membrane_threshold']:.3f}, accuracy={best_config['accuracy']:.4f}")
    
    # Pivot the data to get a 2D matrix for the heatmap
    try:
        heatmap_data = df.pivot(
            index='membrane_threshold', 
            columns='small_world_p', 
            values='accuracy'
        )
        
        print("\n📊 Heatmap Data Preview:")
        print(heatmap_data.to_string())
        print(f"\nShape: {heatmap_data.shape}")
        print(f"Non-NaN values: {heatmap_data.notna().sum().sum()}/{heatmap_data.size}")
        
    except Exception as e:
        print(f"Error pivoting data: {e}")
        print("Raw results:")
        print(df)
        return

    # 5. Plot the heatmap
    plt.figure(figsize=(12, 10))
    
    # Create mask for invalid configurations only
    mask = heatmap_data.isna()
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data, 
        annot=True,       
        fmt=".4f",        
        cmap="viridis",    # Better color scheme: Red=low, Yellow=mid, Green=high
        cbar_kws={'label': 'Test Accuracy'},
        vmin=0.60,        # Adjust to your actual range for better contrast
        vmax=0.75,
        mask=mask,        # Only mask NaN values
        annot_kws={"size": 9, "weight": "bold", "color": "black"},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel("small_world_p  ('beta')", fontsize=14, fontweight='bold')
    plt.ylabel("membrane_threshold  ('theta')", fontsize=14, fontweight='bold')
    plt.title("LSM Accuracy: Effect of Threshold (theta) vs. Rewiring (beta)\n" + 
              f"Theta range: {threshold_range[0]:.2f} to {threshold_range[-1]:.2f} " +
              f"(0.8×I_eff to 10×I_eff, I_eff={I_eff:.3f})", fontsize=16, pad=20)
    
    # Format y-axis labels (theta values) with 2 decimal places
    y_labels = [f"{val:.2f}" for val in heatmap_data.index]
    ax.set_yticklabels(y_labels, rotation=0, fontsize=10)
    
    # Format x-axis labels (beta values) with 2 decimal places
    x_labels = [f"{val:.2f}" for val in heatmap_data.columns]
    ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heatmap_beta_vs_theta_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Heatmap saved to '{filename}'")
    
    # Also save the raw data
    csv_filename = f"grid_search_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ Raw data saved to '{csv_filename}'")
    
    plt.show()

if __name__ == "__main__":
    run_grid_search()