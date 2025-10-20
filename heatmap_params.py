import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas as pd
from datetime import datetime

# Fixed parameters
NUM_OUTPUT_NEURONS = 400
SMALL_WORLD_P = 0.2
RANDOM_SEED = 42

# Grid search parameter ranges
PARAM_GRID = {
    'num_neurons': [750, 1000, 1250, 1500, 2000, 2500, 3000],
    'beta': [25, 50, 75, 100, 125, 150, 200],  # SMALL_WORLD_K / 2
    'tau_ref': [2, 3, 4, 5, 6, 8, 10],  # REFRACTORY_PERIOD
    'theta': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]  # MEMBRANE_THRESHOLD
}

np.random.seed(RANDOM_SEED)

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the preprocessed spike train dataset."""
    print(f"Loading spike train dataset from '{filename}'...")
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples.")
    return X_spikes, y_labels

def calculate_critical_weight(I, beta, theta, tau_ref):
    """Calculate the critical weight for the LSM."""
    w_critico = (theta - 2 * I * tau_ref) / beta
    return w_critico

def extract_features(lsm, spike_data, features_to_extract=["mean_spike_times", "spike_counts"]):
    """Extract features from spike trains using the LSM."""
    all_combined_features = []
    
    for sample in spike_data:
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        
        feature_dict = lsm.extract_features_from_spikes()
        
        sample_feature_parts = []
        for feature_name in features_to_extract:
            feature_vector = feature_dict[feature_name].copy()
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            feature_vector[feature_vector < 0] = 0
            sample_feature_parts.append(feature_vector)
        
        combined_features = np.concatenate(sample_feature_parts)
        all_combined_features.append(combined_features)

    return np.array(all_combined_features)

def evaluate_configuration(X_train, X_test, y_train, y_test, I, num_neurons, beta, tau_ref, theta):
    """Evaluate a single parameter configuration."""
    try:
        # Calculate weight
        small_world_k = int(beta * 2)
        weights_mean = calculate_critical_weight(I, beta, theta, tau_ref)
        
        if weights_mean <= 0:
            return 0.0  # Invalid configuration
        
        # Create LSM
        params = SimulationParams(
            num_neurons=num_neurons,
            mean_weight=weights_mean,
            weight_variance=weights_mean * 5,
            num_output_neurons=NUM_OUTPUT_NEURONS,
            is_random_uniform=False,
            membrane_threshold=theta,
            leak_coefficient=0,
            refractory_period=tau_ref,
            small_world_graph_p=SMALL_WORLD_P,
            small_world_graph_k=small_world_k,
            input_spike_times=X_train[0]
        )
        lsm = SNN(simulation_params=params)
        
        # Extract features
        X_train_features = extract_features(lsm, X_train)
        X_test_features = extract_features(lsm, X_test)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_train_features, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    except Exception as e:
        print(f"Error with config (neurons={num_neurons}, beta={beta}, tau_ref={tau_ref}, theta={theta}): {e}")
        return 0.0

def plot_heatmaps(results_df, save_prefix="lsm_grid_search"):
    """Generate heatmaps for each pair of parameters."""
    params = ['num_neurons', 'beta', 'tau_ref', 'theta']
    param_labels = {
        'num_neurons': 'Number of Neurons',
        'beta': 'Beta (K/2)',
        'tau_ref': 'Refractory Period (τ_ref)',
        'theta': 'Membrane Threshold (θ)'
    }
    
    # Create heatmaps for each parameter pair
    pairs = [
        ('num_neurons', 'beta'),
        ('num_neurons', 'tau_ref'),
        ('num_neurons', 'theta'),
        ('beta', 'tau_ref'),
        ('beta', 'theta'),
        ('tau_ref', 'theta')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (param1, param2) in enumerate(pairs):
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values='accuracy',
            index=param2,
            columns=param1,
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_data,
            ax=axes[idx],
            cmap='RdYlGn',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Accuracy'},
            vmin=0,
            vmax=1
        )
        axes[idx].set_title(f'{param_labels[param1]} vs {param_labels[param2]}')
        axes[idx].set_xlabel(param_labels[param1])
        axes[idx].set_ylabel(param_labels[param2])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"✅ Heatmaps saved to '{save_prefix}_heatmaps.png'")
    plt.show()

def plot_individual_parameter_effects(results_df, save_prefix="lsm_grid_search"):
    """Plot how accuracy varies with each individual parameter."""
    params = ['num_neurons', 'beta', 'tau_ref', 'theta']
    param_labels = {
        'num_neurons': 'Number of Neurons',
        'beta': 'Beta (K/2)',
        'tau_ref': 'Refractory Period (τ_ref)',
        'theta': 'Membrane Threshold (θ)'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(params):
        param_means = results_df.groupby(param)['accuracy'].agg(['mean', 'std', 'count'])
        
        axes[idx].errorbar(
            param_means.index,
            param_means['mean'],
            yerr=param_means['std'],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            capsize=5
        )
        axes[idx].set_xlabel(param_labels[param], fontsize=12)
        axes[idx].set_ylabel('Accuracy', fontsize=12)
        axes[idx].set_title(f'Effect of {param_labels[param]} on Accuracy', fontsize=14)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_individual_effects.png', dpi=300, bbox_inches='tight')
    print(f"✅ Individual parameter effects saved to '{save_prefix}_individual_effects.png'")
    plt.show()

def main():
    # Load dataset
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    # Calculate input spike rate
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    print(f"Input spike rate (I): {I:.6f}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_labels
    )
    print(f"Split: {len(X_train)} training, {len(X_test)} test samples.\n")
    
    # Generate all parameter combinations
    param_combinations = list(product(
        PARAM_GRID['num_neurons'],
        PARAM_GRID['beta'],
        PARAM_GRID['tau_ref'],
        PARAM_GRID['theta']
    ))
    
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to test: {total_combinations}\n")
    print("="*70)
    
    # Store results
    results = []
    
    # Run grid search
    for i, (num_neurons, beta, tau_ref, theta) in enumerate(tqdm(param_combinations, desc="Grid Search Progress")):
        accuracy = evaluate_configuration(X_train, X_test, y_train, y_test, I, num_neurons, beta, tau_ref, theta)
        
        results.append({
            'num_neurons': num_neurons,
            'beta': beta,
            'tau_ref': tau_ref,
            'theta': theta,
            'accuracy': accuracy
        })
        
        # Save intermediate results every 50 iterations
        if (i + 1) % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f'lsm_grid_search_intermediate_{i+1}.csv', index=False)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'lsm_grid_search_results_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\n✅ Results saved to '{results_filename}'")
    
    # Find best configuration
    best_idx = results_df['accuracy'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND:")
    print("="*70)
    print(f"Number of Neurons: {best_config['num_neurons']}")
    print(f"Beta (K/2): {best_config['beta']}")
    print(f"Refractory Period (τ_ref): {best_config['tau_ref']}")
    print(f"Membrane Threshold (θ): {best_config['theta']}")
    print(f"Accuracy: {best_config['accuracy']*100:.2f}%")
    print("="*70)
    
    # Print top 10 configurations
    print("\nTOP 10 CONFIGURATIONS:")
    print("-"*70)
    top_10 = results_df.nlargest(10, 'accuracy')
    print(top_10.to_string(index=False))
    print("-"*70)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_heatmaps(results_df, save_prefix=f'lsm_grid_search_{timestamp}')
    plot_individual_parameter_effects(results_df, save_prefix=f'lsm_grid_search_{timestamp}')
    
    print("\n✅ Grid search complete!")

if __name__ == "__main__":
    main()