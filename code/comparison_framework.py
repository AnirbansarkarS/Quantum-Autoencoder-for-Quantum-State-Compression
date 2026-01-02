import numpy as np
from qiskit.quantum_info import state_fidelity, Statevector
import json
import os

class ComparisonFramework:
    """
    Standardizes experiments and metrics for comparing Quantum and Classical 
    Autoencoders on quantum state data.
    """
    
    def __init__(self, results_dir='results/comparisons'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.registry = {}

    def calculate_fidelity(self, target_state, predicted_state):
        """
        Computes quantum fidelity between two states.
        Both should be Statevectors or numpy arrays representing statevectors.
        """
        return state_fidelity(target_state, predicted_state)

    def log_result(self, method_name, qubit_count, latent_count, fidelity, training_time, status='Success'):
        """
        Logs a specific experiment result.
        """
        result = {
            "method": method_name,
            "qubits": qubit_count,
            "latent": latent_count,
            "fidelity": float(fidelity),
            "training_time": float(training_time),
            "status": status
        }
        
        file_name = f"{method_name}_q{qubit_count}_l{latent_count}.json"
        save_path = os.path.join(self.results_dir, file_name)
        
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        print(f"Logged {method_name} result to {save_path}")

    def get_summary_table(self):
        """
        Reads all logged results and returns a summary.
        """
        results = []
        for file in os.listdir(self.results_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    results.append(json.load(f))
        return results

# Example Usage:
# framework = ComparisonFramework()
# framework.log_result("QAE", 4, 1, 0.99, 120.5)
# framework.log_result("Classical", 4, 1, 0.45, 10.2)
