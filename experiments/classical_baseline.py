import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from qiskit.quantum_info import Statevector, state_fidelity

# Add both 'code' and its parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))
from comparison_framework import ComparisonFramework

class ClassicalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ClassicalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_classical_baseline():
    framework = ComparisonFramework(results_dir='results/experiments')
    qubit_range = [2, 4, 6, 8, 10, 12]
    fidelities = []
    
    # We fix the latent dimension to a small number (e.g., 32 real features)
    # This simulates the "bottleneck" where information MUST be squeezed
    LATENT_DIM = 16 

    for n_q in qubit_range:
        print(f"\n--- Testing Classical Baseline for {n_q} Qubits ---")
        input_dim = 2 * (2**n_q)
        
        # Prepare a dataset of 50 random quantum states
        dataset = []
        for _ in range(50):
            v = np.random.randn(2**n_q) + 1j * np.random.randn(2**n_q)
            v /= np.linalg.norm(v)
            dataset.append(torch.cat([torch.tensor(v.real), torch.tensor(v.imag)]).float())
        
        X_train = torch.stack(dataset)
        
        model = ClassicalAutoencoder(input_dim, LATENT_DIM)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(300):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, X_train)
            loss.backward()
            optimizer.step()
        
        # Test on a NEW random state
        v_test = np.random.randn(2**n_q) + 1j * np.random.randn(2**n_q)
        v_test /= np.linalg.norm(v_test)
        x_test = torch.cat([torch.tensor(v_test.real), torch.tensor(v_test.imag)]).float().unsqueeze(0)
        
        with torch.no_grad():
            reconstructed_vec = model(x_test).squeeze(0).numpy()
            n = len(reconstructed_vec) // 2
            reconstructed_complex = reconstructed_vec[:n] + 1j * reconstructed_vec[n:]
            # Normalize
            reconstructed_complex /= np.linalg.norm(reconstructed_complex)
            
            fid = state_fidelity(v_test, reconstructed_complex)
            fidelities.append(fid)
            print(f"Test Fidelity for {n_q} qubits (Latent={LATENT_DIM}): {fid:.6f}")
            framework.log_result("Classical_Generalization", n_q, LATENT_DIM, fid, 0)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_range, fidelities, marker='o', color='red', label=f'Classical AE (Latent={LATENT_DIM})')
    plt.axhline(y=0.9, color='green', linestyle='--', label='Target Threshold')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Test Reconstruction Fidelity')
    plt.title('The Kill Shot: Classical Generalization Collapse')
    plt.legend()
    plt.grid(True)
    
    save_path = 'results/experiments/classical_collapse.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nCollapse Plot saved to {save_path}")

if __name__ == "__main__":
    run_classical_baseline()
