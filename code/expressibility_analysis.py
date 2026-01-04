import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

from train_qae import train_qae
from comparison_framework import ComparisonFramework

def run_expressibility_study():
    """
    Analyzes how the depth of the variational ansatz (reps) affects 
    the compression fidelity of complex quantum states.
    """
    framework = ComparisonFramework(results_dir='results/experiments')
    results = []
    
    # 1. State Preparation (4-qubit random state)
    n_q = 4
    n_l = 2
    
    print(f"\n--- Starting Expressibility Study ({n_q} qubits -> {n_l} latent) ---")
    
    # Prepare a complex random state with significant entanglement
    random_vec = np.random.randn(2**n_q) + 1j * np.random.randn(2**n_q)
    random_vec /= np.linalg.norm(random_vec)
    input_sv = Statevector(random_vec)
    
    reps_to_test = [1, 2, 3, 4, 5, 8]
    final_fidelities = []
    
    for reps in reps_to_test:
        print(f"\nTesting Depth (Reps): {reps}")
        
        # We need a way to pass 'reps' to train_qae. 
        # Since train_qae uses QuantumAutoencoder(num_qubits, num_latent)
        # which defaults to reps=3, we'll temporarily import and use the class directly
        from quantum_autoencoder import QuantumAutoencoder
        from loss_functions import trash_population_loss
        from qiskit_algorithms.optimizers import COBYLA
        
        qae = QuantumAutoencoder(num_qubits=n_q, num_latent=n_l, reps=reps)
        num_params = qae.num_parameters
        initial_params = np.random.uniform(0, 2 * np.pi, num_params)
        
        def objective(params):
            qc = qae.get_autoencoder_circuit(params)
            final_state = input_sv.evolve(qc)
            return trash_population_loss(final_state, qae.num_trash)
        
        optimizer = COBYLA(maxiter=100)
        res = optimizer.minimize(objective, initial_params)
        
        fidelity = 1 - res.fun
        final_fidelities.append(fidelity)
        print(f"Final Fidelity for Reps={reps}: {fidelity:.6f}")
        
        framework.log_result(f"Expressibility_Reps{reps}", n_q, n_l, fidelity, 0)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(reps_to_test, final_fidelities, marker='o', linestyle='-', color='purple')
    plt.xlabel('Anstaz Depth (Reps)')
    plt.ylabel('Compression Fidelity')
    plt.title('Expressibility Study: Fidelity vs. Circuit Depth')
    plt.grid(True)
    
    save_path = 'results/experiments/expressibility_depth.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nExpressibility Plot saved to {save_path}")

if __name__ == "__main__":
    run_expressibility_study()
