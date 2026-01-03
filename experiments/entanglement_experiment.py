import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Add both 'code' and its parent to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

from train_qae import train_qae
from comparison_framework import ComparisonFramework

def run_entanglement_experiment():
    framework = ComparisonFramework(results_dir='results/experiments')
    
    # 1. Bell State Experiment (2 qubits -> 1 latent)
    print("\n--- Running Bell State Experiment ---")
    n_q, n_l = 2, 1
    qc_bell = QuantumCircuit(n_q)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    sv_bell = Statevector.from_instruction(qc_bell)
    
    params_bell, history_bell = train_qae(n_q, n_l, sv_bell, epochs=100)
    framework.log_result("QAE_Bell", n_q, n_l, 1 - history_bell[-1], 0) # Time placeholder

    # 2. GHZ State Experiment (3 qubits -> 1 latent)
    print("\n--- Running GHZ State Experiment ---")
    n_q, n_l = 3, 1
    qc_ghz = QuantumCircuit(n_q)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(0, 2)
    sv_ghz = Statevector.from_instruction(qc_ghz)
    
    params_ghz, history_ghz = train_qae(n_q, n_l, sv_ghz, epochs=150)
    framework.log_result("QAE_GHZ", n_q, n_l, 1 - history_ghz[-1], 0)

    # 3. Random Product State (3 qubits -> 1 latent)
    print("\n--- Running Random Product State Experiment ---")
    n_q, n_l = 3, 1
    qc_prod = QuantumCircuit(n_q)
    qc_prod.h(0) # Some random-ish state
    qc_prod.h(1)
    qc_prod.h(2)
    sv_prod = Statevector.from_instruction(qc_prod)
    
    params_prod, history_prod = train_qae(n_q, n_l, sv_prod, epochs=150)
    framework.log_result("QAE_Product", n_q, n_l, 1 - history_prod[-1], 0)

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(history_bell, label='Bell State (n=2, l=1)', linestyle='--')
    plt.plot(history_ghz, label='GHZ State (n=3, l=1)', linestyle='-')
    plt.plot(history_prod, label='Product State (n=3, l=1)', linestyle=':')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (1 - P(trash=|0>))')
    plt.title('Quantum Autoencoder: Entanglement vs Product State Compression')
    plt.legend()
    plt.grid(True)
    
    save_path = 'results/experiments/entanglement_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nMetric Plot saved to {save_path}")

if __name__ == "__main__":
    run_entanglement_experiment()
