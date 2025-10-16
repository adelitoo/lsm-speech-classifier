"""
Corrected Liquid State Machine using Nengo
Matches the snnpy implementation architecture
"""

import numpy as np
import nengo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from tqdm import tqdm

# Parameters matching original implementation
NUM_NEURONS = 3000
NUM_OUTPUT_NEURONS = 600
REFRACTORY_PERIOD = 0.007  # 7ms
SIMULATION_TIME = 0.1  # 100ms
DT = 0.001  # 1ms timestep
N_MELS = 200
TIME_BINS = 100

class NengoLSM:
    """Liquid State Machine with correct spike-based implementation"""
    
    def __init__(self, n_inputs=200, n_reservoir=3000, n_readout=600, 
                 k=200, p=0.3, threshold=2.0):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_readout = n_readout
        self.dt = DT
        self.k = k  # Small-world parameter
        self.p = p  # Rewiring probability
        
        self.current_input_spikes = None
        
    def _create_small_world_connections(self):
        """Create small-world connectivity matrix (Watts-Strogatz)"""
        n = self.n_reservoir
        k = self.k
        p = self.p
        
        # Start with ring lattice
        connections = np.zeros((n, n))
        for i in range(n):
            for j in range(1, k // 2 + 1):
                connections[i, (i + j) % n] = 1
                connections[i, (i - j) % n] = 1
        
        # Rewire edges with probability p
        for i in range(n):
            for j in range(i + 1, n):
                if connections[i, j] == 1 and np.random.rand() < p:
                    # Remove edge
                    connections[i, j] = 0
                    # Add random edge
                    new_target = np.random.randint(n)
                    while new_target == i or connections[i, new_target] == 1:
                        new_target = np.random.randint(n)
                    connections[i, new_target] = 1
        
        return connections
    
    def build_network(self, weights_mean, weights_std):
        """Build the LSM network"""
        self.model = nengo.Network()
        
        with self.model:
            # Input layer - spike generators for each mel band
            self.input_nodes = []
            for i in range(self.n_inputs):
                node = nengo.Node(self._spike_generator(i))
                self.input_nodes.append(node)
            
            # Reservoir - independent spiking neurons
            self.reservoir = nengo.Ensemble(
                n_neurons=self.n_reservoir,
                dimensions=1,  # 1D space, neurons are just independent
                neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=REFRACTORY_PERIOD),
                max_rates=nengo.dists.Choice([100]),  # Fixed rate
                intercepts=nengo.dists.Choice([0]),
                encoders=nengo.dists.Choice([[1]])  # All encode same dimension
            )
            
            # Input to reservoir connections (random)
            input_to_res_weights = np.random.normal(
                weights_mean, weights_std, 
                (self.n_reservoir, self.n_inputs)
            )
            
            # Connect each input to reservoir
            for i, input_node in enumerate(self.input_nodes):
                nengo.Connection(
                    input_node,
                    self.reservoir.neurons,
                    transform=input_to_res_weights[:, i:i+1],
                    synapse=None
                )
            
            # Recurrent connections (small-world)
            topology = self._create_small_world_connections()
            recurrent_weights = np.random.normal(
                weights_mean, weights_std,
                (self.n_reservoir, self.n_reservoir)
            ) * topology
            
            nengo.Connection(
                self.reservoir.neurons,
                self.reservoir.neurons,
                transform=recurrent_weights,
                synapse=0.005
            )
            
            # Readout layer
            self.readout = nengo.Ensemble(
                n_neurons=self.n_readout,
                dimensions=1,
                neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=REFRACTORY_PERIOD),
                max_rates=nengo.dists.Choice([100]),
                intercepts=nengo.dists.Choice([0]),
                encoders=nengo.dists.Choice([[1]])
            )
            
            # Reservoir to readout
            res_to_read_weights = np.random.normal(
                weights_mean * 0.5, weights_std * 0.5,
                (self.n_readout, self.n_reservoir)
            )
            
            nengo.Connection(
                self.reservoir.neurons,
                self.readout.neurons,
                transform=res_to_read_weights,
                synapse=0.005
            )
            
            # Probe readout spikes
            self.spike_probe = nengo.Probe(self.readout.neurons)
    
    def _spike_generator(self, neuron_idx):
        """Generate spike function for a specific input neuron"""
        def spike_func(t):
            if self.current_input_spikes is None:
                return 0
            
            time_bin = int(t / self.dt)
            if time_bin >= TIME_BINS:
                return 0
            
            # Return large current when spike occurs
            return 10.0 if self.current_input_spikes[neuron_idx, time_bin] else 0.0
        
        return spike_func
    
    def build_simulator(self):
        """Build simulator once"""
        self.sim = nengo.Simulator(self.model, dt=self.dt)
    
    def process_sample(self, spike_train):
        """Process one sample through the LSM"""
        self.current_input_spikes = spike_train
        self.sim.reset()
        self.sim.run(SIMULATION_TIME)
        
        # Extract spike data
        spike_data = self.sim.data[self.spike_probe]
        
        # Calculate mean spike time for each readout neuron
        features = self._extract_mean_spike_times(spike_data)
        return features
    
    def _extract_mean_spike_times(self, spike_data):
        """Extract mean spike time for each neuron"""
        n_neurons = spike_data.shape[1]
        mean_times = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            # Find where spikes occur (spike_data is already binary in neuron probes)
            spike_indices = np.where(spike_data[:, i] > 0)[0]
            
            if len(spike_indices) > 0:
                # Mean time in ms
                mean_times[i] = np.mean(spike_indices) * self.dt * 1000
            else:
                mean_times[i] = 0
        
        # Set negative values to 0 (matching original)
        mean_times[mean_times < 0] = 0
        
        return mean_times
    
    def close(self):
        """Clean up"""
        if hasattr(self, 'sim'):
            self.sim.close()


def load_spike_dataset(filename="speech_spike_dataset.npz"):
    """Load the spike dataset"""
    print(f"Loading spike train dataset from '{filename}'...")
    if not Path(filename).exists():
        print(f"Error: Dataset file not found at '{filename}'")
        return None, None
    
    data = np.load(filename)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples.")
    return X_spikes, y_labels


def calculate_critical_weight(X_spikes, k=200, threshold=2.0, ref_period=7):
    """Calculate critical weight matching snnpy implementation"""
    num_samples, num_inputs, time_bins = X_spikes.shape
    I = np.sum(X_spikes) / (num_samples * num_inputs * time_bins)
    beta = k / 2
    w_critico = (threshold - 2 * I * ref_period) / beta
    return w_critico


def extract_features(lsm, spike_data, desc=""):
    """Extract features from spike trains"""
    features = []
    all_means = []
    all_maxes = []
    
    for sample in tqdm(spike_data, desc=desc):
        feature_vec = lsm.process_sample(sample)
        features.append(feature_vec)
        
        if np.any(feature_vec > 0):
            all_means.append(np.mean(feature_vec[feature_vec > 0]))
        all_maxes.append(np.max(feature_vec))
    
    print(f"\n--- Feature Statistics ({desc}) ---")
    if all_means:
        print(f"  Avg of mean spike times: {np.mean(all_means):.2f}")
    print(f"  Overall max spike time: {np.max(all_maxes):.2f}")
    
    return np.array(features)


def main():
    # Load data
    X_spikes, y_labels = load_spike_dataset()
    if X_spikes is None:
        return
    
    # Calculate critical weight
    w_critico = calculate_critical_weight(X_spikes)
    print(f"\nCalculated Critical Weight: {w_critico:.6f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels
    )
    print(f"Split: {len(X_train)} train, {len(X_test)} test")
    
    # Build LSM
    print("\nBuilding Nengo LSM...")
    lsm = NengoLSM(
        n_inputs=N_MELS,
        n_reservoir=NUM_NEURONS,
        n_readout=NUM_OUTPUT_NEURONS,
        k=200,
        p=0.3,
        threshold=2.0
    )
    
    lsm.build_network(weights_mean=w_critico, weights_std=w_critico * 5)
    print("Building simulator...")
    lsm.build_simulator()
    print("✅ LSM ready")
    
    # Extract features
    print("\nExtracting training features...")
    X_train_features = extract_features(lsm, X_train, "Training")
    
    print("\nExtracting test features...")
    X_test_features = extract_features(lsm, X_test, "Test")
    
    # Train classifier
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train_features, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["yes", "no", "up", "down"]
    )
    
    print("\n" + "="*50)
    print("  RESULTS (Corrected Nengo LSM)")
    print("="*50)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
    print(report)
    
    # Save
    np.savez_compressed(
        "nengo_lsm_corrected_features.npz",
        X_train_features=X_train_features,
        y_train=y_train,
        X_test_features=X_test_features,
        y_test=y_test
    )
    print("\n✅ Complete!")
    
    lsm.close()


if __name__ == "__main__":
    main()