import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA, SPSA
from quantum_autoencoder import QuantumAutoencoder
from loss_functions import trash_population_loss
import os

def train_qae(n_qubits, n_latent, input_state, epochs=100, optimizer_type='COBYLA'):
    """
    Core training loop for the Quantum Autoencoder.
    """
    qae = QuantumAutoencoder(num_qubits=n_qubits, num_latent=n_latent)
    num_params = qae.num_parameters
    
    # Initialize random parameters
    initial_params = np.random.uniform(0, 2 * np.pi, num_params)
    
    loss_history = []
    
    def objective_function(params):
        # 1. Build circuit with current parameters
        qc = qae.get_autoencoder_circuit(params)
        
        # 2. Evolve the input state
        # In a real QAE, you'd apply the circuit to the input state
        final_state = input_state.evolve(qc)
        
        # 3. Calculate loss (trash qubit population)
        loss = trash_population_loss(final_state, qae.num_trash)
        
        loss_history.append(loss)
        return loss
    
    # Setup optimizer
    if optimizer_type == 'COBYLA':
        optimizer = COBYLA(maxiter=epochs)
    elif optimizer_type == 'SPSA':
        optimizer = SPSA(maxiter=epochs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Run optimization
    print(f"Starting training with {optimizer_type} for {epochs} iterations...")
    result = optimizer.minimize(objective_function, initial_params)
    
    optimal_params = result.x
    final_loss = result.fun
    
    print(f"Training complete. Final Loss: {final_loss:.6f}")
    
    return optimal_params, loss_history

def plot_loss(loss_history, save_path='results/loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Trash Population Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (1 - P(trash=|0>))')
    plt.title('QAE Training Progress')
    plt.legend()
    plt.grid(True)
    
    # Create results dir if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example Run: Compress a Bell state (2 qubits -> 1 latent)
    # Note: Bell states are entangled.
    from qiskit import QuantumCircuit
    
    n_q = 2
    n_l = 1
    
    # Prepare Bell state |phi+> = (|00> + |11>)/sqrt(2)
    bell_qc = QuantumCircuit(n_q)
    bell_qc.h(0)
    bell_qc.cx(0, 1)
    input_sv = Statevector(bell_qc)
    
    opt_params, history = train_qae(n_q, n_l, input_sv, epochs=50)
    plot_loss(history, save_path='results/bell_state_compression.png')
