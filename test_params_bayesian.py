import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import optuna
from datetime import datetime
import json
import pandas as pd

# Parameter search spaces (min, max, step for integers; min, max for floats)
PARAM_SEARCH_SPACE = {
    'num_neurons': (1000, 5000, 200),  # min, max, step
    'num_output_neurons': (300, 1000, 100),
    'refractory_period': (1, 12, 1),
    'membrane_threshold': (1.0, 5.0),  # continuous
    'small_world_p': (0.1, 0.6),  # continuous
    'small_world_k': (100, 500, 50),  # will be made even
}

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'leak_coefficient': 0
}

# Global variables to store data (to avoid reloading)
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None

def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike train dataset."""
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    data = np.load(filename)
    return data['X_spikes'], data['y_labels']

def extract_features(lsm, spike_data, desc=""):
    """Extract mean spike time features from LSM."""
    features = []
    for sample in tqdm(spike_data, desc=desc, leave=False):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate()
        feature_dict = lsm.extract_features_from_spikes()
        current_features = feature_dict["mean_spike_times"]
        current_features[current_features < 0] = 0
        features.append(current_features)
    return np.array(features)

def calculate_critical_weight(X_spikes, config):
    """Calculate critical weight for the given configuration."""
    num_samples = X_spikes.shape[0]
    num_input_neurons = X_spikes.shape[1]
    num_time_bins = X_spikes.shape[2]
    
    I = np.sum(X_spikes) / (num_samples * num_input_neurons * num_time_bins)
    beta = config['small_world_k'] / 2
    w_critico = (config['membrane_threshold'] - 2 * I * config['refractory_period']) / beta
    return w_critico

def objective(trial):
    """
    Objective function for Optuna optimization.
    Returns accuracy to MAXIMIZE.
    """
    # Sample parameters from the search space
    config = {
        'num_neurons': trial.suggest_int('num_neurons', *PARAM_SEARCH_SPACE['num_neurons']),
        'num_output_neurons': trial.suggest_int('num_output_neurons', *PARAM_SEARCH_SPACE['num_output_neurons']),
        'refractory_period': trial.suggest_int('refractory_period', *PARAM_SEARCH_SPACE['refractory_period']),
        'membrane_threshold': trial.suggest_float('membrane_threshold', *PARAM_SEARCH_SPACE['membrane_threshold']),
        'small_world_p': trial.suggest_float('small_world_p', *PARAM_SEARCH_SPACE['small_world_p']),
    }
    
    # Add fixed parameters
    config.update(FIXED_PARAMS)
    
    # Sample k and ensure it's even
    k_raw = trial.suggest_int('small_world_k_raw', *PARAM_SEARCH_SPACE['small_world_k'])
    config['small_world_k'] = k_raw if k_raw % 2 == 0 else k_raw + 1
    
    try:
        # Calculate critical weight for this configuration
        X_full = np.concatenate([X_TRAIN, X_TEST], axis=0)
        weights_mean = calculate_critical_weight(X_full, config)
        
        # Create SNN with current configuration
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
            input_spike_times=X_TRAIN[0]
        )
        
        lsm = SNN(simulation_params=params)
        
        # Extract features
        X_train_features = extract_features(lsm, X_TRAIN, desc=f"Trial {trial.number}")
        X_test_features = extract_features(lsm, X_TEST, desc=f"Trial {trial.number}")
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train_features, Y_TRAIN)
        
        # Evaluate
        y_pred = clf.predict(X_test_features)
        accuracy = accuracy_score(Y_TEST, y_pred)
        
        # Store critical weight as user attribute
        trial.set_user_attr('critical_weight', weights_mean)
        
        print(f"\nTrial {trial.number}: Accuracy = {accuracy*100:.2f}%")
        
        return accuracy
    
    except Exception as e:
        print(f"\nTrial {trial.number} failed: {e}")
        # Return a very low accuracy for failed trials
        return 0.0

def run_bayesian_optimization(n_trials=50, timeout=None):
    """
    Run Bayesian optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (None = no limit)
    """
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    
    print("="*70)
    print("LSM BAYESIAN OPTIMIZATION (Optuna)")
    print("="*70)
    
    # Load data once
    print("\nLoading dataset...")
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Dataset loaded: {len(X_TRAIN)} train, {len(X_TEST)} test samples")
    
    print(f"\nStarting Bayesian optimization with {n_trials} trials...")
    print("Optuna will intelligently search for optimal parameters.\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # We want to maximize accuracy
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
        study_name=f'lsm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value*100:.2f}%")
    
    print("\nBest Parameters:")
    for param, value in study.best_params.items():
        if param == 'small_world_k_raw':
            actual_k = value if value % 2 == 0 else value + 1
            print(f"  small_world_k: {actual_k}")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nCritical Weight: {study.best_trial.user_attrs['critical_weight']:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study
    study_filename = f"lsm_bayesian_study_{timestamp}.pkl"
    optuna.save_study(study, study_filename)
    print(f"\nStudy saved to: {study_filename}")
    
    # Save best parameters to JSON
    best_params = study.best_params.copy()
    if 'small_world_k_raw' in best_params:
        k_raw = best_params.pop('small_world_k_raw')
        best_params['small_world_k'] = k_raw if k_raw % 2 == 0 else k_raw + 1
    
    best_config = {
        'best_accuracy': study.best_value,
        'best_trial': study.best_trial.number,
        'critical_weight': study.best_trial.user_attrs['critical_weight'],
        'parameters': best_params
    }
    
    json_filename = f"lsm_best_config_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Best configuration saved to: {json_filename}")
    
    # Save all trials to CSV
    import pandas as pd
    trials_data = []
    for trial in study.trials:
        if trial.value is None:
            continue
        
        trial_dict = {
            'trial_number': trial.number,
            'accuracy': trial.value,
            'critical_weight': trial.user_attrs.get('critical_weight', None)
        }
        
        # Add parameters
        for param, value in trial.params.items():
            if param == 'small_world_k_raw':
                trial_dict['small_world_k'] = value if value % 2 == 0 else value + 1
            else:
                trial_dict[param] = value
        
        # Add fixed parameters
        trial_dict['leak_coefficient'] = FIXED_PARAMS['leak_coefficient']
        
        trials_data.append(trial_dict)
    
    df = pd.DataFrame(trials_data)
    csv_filename = f"lsm_bayesian_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"All trial results saved to: {csv_filename}")
    
    # Top 5 trials
    print("\n" + "="*70)
    print("TOP 5 TRIALS")
    print("="*70)
    
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for i, trial in enumerate(top_trials, 1):
        if trial.value is None:
            continue
        print(f"\n#{i} - Trial {trial.number}: Accuracy = {trial.value*100:.2f}%")
        print(f"  Parameters:")
        for param, value in trial.params.items():
            if param == 'small_world_k_raw':
                actual_k = value if value % 2 == 0 else value + 1
                print(f"    small_world_k: {actual_k}")
            else:
                print(f"    {param}: {value}")
    
    # Parameter importance
    print("\n" + "="*70)
    print("PARAMETER IMPORTANCE")
    print("="*70)
    print("\n(How much each parameter affects accuracy)")
    
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            if param == 'small_world_k_raw':
                param = 'small_world_k'
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  Could not calculate importance: {e}")
    
    # Optimization history
    print("\n" + "="*70)
    print("OPTIMIZATION PROGRESS")
    print("="*70)
    
    accuracies = [t.value*100 for t in study.trials if t.value is not None]
    if accuracies:
        print(f"\nFirst trial accuracy: {accuracies[0]:.2f}%")
        print(f"Best accuracy found: {max(accuracies):.2f}%")
        print(f"Improvement: +{max(accuracies) - accuracies[0]:.2f}%")
    
    return study

def continue_optimization(study_filename, n_additional_trials=20):
    """
    Continue optimization from a saved study.
    
    Args:
        study_filename: Path to saved study (.pkl file)
        n_additional_trials: Number of additional trials to run
    """
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    
    print(f"Loading study from {study_filename}...")
    study = optuna.load_study(study_filename)
    
    print(f"Continuing optimization with {n_additional_trials} additional trials...")
    print(f"Current best accuracy: {study.best_value*100:.2f}%")
    
    # Load data
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Continue optimization
    study.optimize(objective, n_trials=n_additional_trials, show_progress_bar=True)
    
    print(f"\nNew best accuracy: {study.best_value*100:.2f}%")
    
    return study

def main():
    """Main function."""
    
    # Run Bayesian optimization
    study = run_bayesian_optimization(n_trials=100)
    
    # Optional: Continue optimization if you want more trials
    # study = continue_optimization('lsm_bayesian_study_YYYYMMDD_HHMMSS.pkl', n_additional_trials=20)
    
    # Optional: Create visualization (requires plotly)
    # try:
    #     import plotly
    #     fig = optuna.visualization.plot_optimization_history(study)
    #     fig.write_html('optimization_history.html')
    #     print("\nVisualization saved to: optimization_history.html")
    # except ImportError:
    #     print("\nInstall plotly to generate visualizations: pip install plotly")

if __name__ == "__main__":
    main()